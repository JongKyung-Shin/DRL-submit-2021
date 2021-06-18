# DRL-submit-2021 - by JongKyung Shin (20205357)

1. Details
  * Training framework 구동 code
    * main.py : 전체 training framework를 작동 시키는 code
    * Agent.py : REINFORCE policy network code
  * State generator code
    * state_generator.py : CNN network 및 state 생성을 위한 code
  * state 구현을 위한 전처리 code
    * sequence_generator.py : TSP 알고리즘으로 구매 순서 생성
    * preprocessing.ipynb : 도면 이미지 생성
    * preprocessing_Product_by_md_data_generation.ipynb : H matrix 데이터 생성 (각 매대에 어떤 상품이 있는지)
    * preprocessing_data_generation.ipynb : 구매 데이터로 학습을 위한 구매순서있는 transaction생성 (json_file)
    * preprecess_number_of_keys_sperate.ipynb : T 길이만큼의 transaction 추출
 
3. Dependency
  * main code 관련
    * os : Ubuntu 18.04
    * Pytorch
    * matplotlib
    * IPython
    * pandas
    * numpy
    * json
  * 데이터 생성 관련
    * ortools : TSP solver
    * tqdm
