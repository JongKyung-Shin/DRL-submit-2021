from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

def create_data_model(distance_matrix):
    data = {}
    data["distance_matrix"] = distance_matrix
    data["starts"] = [0]
    data["ends"] = [len(distance_matrix)-1]
    data['num_vehicles'] = 1
    return data

def Euclidean_Distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.sqrt(np.sum(np.square(point1- point2)))
    return distance

def Make_Distance_Metrix(start_point : list, path_list : list ,end_point : list):
    node_distance_matrix = []
    all_path_list = [start_point] + path_list + [end_point]
    for point1 in all_path_list:
        temp_list = []
        for point2 in all_path_list:
            temp_list.append(Euclidean_Distance(point1, point2))
        node_distance_matrix.append(temp_list)
    return node_distance_matrix, all_path_list

def change_ordered_location(index_ordered_path_list, all_path_list):
    temp = []
    for idx in index_ordered_path_list:
        temp.append(all_path_list[idx])
    return temp


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    #print(f'Objective: {solution.ObjectiveValue()}')
    max_route_distance = 0
    idx_path_list = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            idx_path_list.append(manager.IndexToNode(index))
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        idx_path_list.append(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        #print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    #print('Maximum of the route distances: {}m'.format(max_route_distance))
    return idx_path_list


#main function
def sequence_generation(start_point,path_list,end_point):

    node_distace_matrix, all_path_list = Make_Distance_Metrix(start_point, path_list, end_point)
    data = create_data_model(node_distace_matrix)
    
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['starts'],data['ends'])
    
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    
    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        2000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        idx_path_list = print_solution(data, manager, routing, solution)
        #ordered_path_list = change_ordered_location(idx_path_list, all_path_list)

    return idx_path_list
