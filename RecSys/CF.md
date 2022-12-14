# Collaborative Filtering (수식 : Recsys 이론 4강 참고)
'많은 유저'들로부터 얻은 기호 정보를 이용해 유저의 관심사를 예측하는 방법

## Neighborhood Based CF(Memory Based CF)

- User-based CF
  '두 유저가 얼마나 유사한 아이템을 선호하는가?'
  - 유저간 유사도를 구한 뒤, 타겟 유저와 유사도가 높은 유저들이 선호하는 아이템 추천

- Item-based CF
  '두 아이템이 유저들로부터 얼마나 유사한 평점을 받았는가?'
  - 아이템간 유사도를 구한 뒤, 타겟 아이템과 유사도가 높은 아이템 중 선호도가 큰 아이템을 추천

- 특징
  - 구현이 간단하고 이해하기 쉽다
  - 아이템이나 유저가 계속 늘어날 경우 확장성이 떨어진다
  - 데이터가 적을 경우, 성능이 저하된다

### K-Nearest Neighbors CF(KNN CF)
- 기본 NBCF는 모든 유사도를 고려해야 한다는 단점이 존재
  - 유저가 많아질 수록, 연산은 늘어나고 성능이 떨어지기도 함
- __일정 범위에 속한 유저__ 가운데 유저1과 가장 유사한 K명의 유저를 이용해 평점 예측
- 유사도 측정법
  - Euclidean Space(Mean Squared Difference)
    - 두 개체 간 (유클리드)거리를 계산
  - Cosine Similarity
    - 두 벡터의 각도를 이용해서 유사도를 계산
  - Jacaard Similarity
    - 집합의 개념을 사용한 유사도
  - Pearson Similarity
    - 상관계수

## Model Based CF
- 항목간 유사성을 단순 비교하는 것에서 벗어나, __데이터에 내재한 패턴을 이용해__ 추천하는 CF기법
- __Latent Factor__
- 장점
  - 모델을 학습시키기 떄문에 모델에 저장된 패턴만 사용하면 되므로 서빙이 빠르다
    - Memory based는 모든 유사도를 계산해야하기 때문에 연산량이 많고 서빙이 느림
  - Sparsity, Scalability 문제 개선
  - Overfitting 방지
  - Limited Coverage 극복
    - Memory based의 경우 공통의 유저, 아이템을 많이 공유해야 유사도 값이 정확해짐

### Singular Value Decomposition(SVD)
- Rating matrix(R)을 유저와 아이템의 잠재 요인을 포함할 수 있는 행렬로 분해 __(3개의 행렬 : user, item, latent matrix)__
- 한계점
  - 분해하려는 행렬의 knowledge가 불완전할 때 정의되지 않음
  - 따라서 imputation을 통해 dense matrix를 임의로 생성 -> 정확하지 않다면 데이터를 왜곡시키고 성능이 저하됨

### Matrix Factorization(MF)
- Rating Matrix(R)을 User, item latent factor matrix로 저차원 분해 시키는 방법 __(2개의 행렬 : user, item)__
- SVD와 유사하지만, 관측된 평점만 모델링에 활용, 학습
- Adding Biases
  - 유저, 아이템에 각각에 대해서 bias가 존재할 수 있음
  - 전체, 유저, 아이템 각각에 대한 bias를 모델에 추가
- Adding Confidence Level
  - 모든 평점이 동일한 신뢰도를 갖지 않으므로 __confidence level term__ 추가
    - 대규모 광고 집행과 같이 특정 아이템이 많이 노출되어 클릭되는 경우
    - implicit feedback
      - + preference
        - 유저가 아이템을 선호하는지 여부를 binary하게 표현
      - + confidence
        - 유저가 아이템을 선호하는 정도를 나타내는 increasing function
- Adding temporal dynamics
  - 시간에 따라 변화하는 유저, 아이템의 특성 반영

### Alternative Least Square(ALS) -> 연산 방식
- 유저와 아이템 행렬을 번갈아가면서 업데이트
- 하나를 고정하고 다른 하나로 OLS문제를 푸는 것
- SGD와의 비교 
  - SGD과 비교해서 sparse Data에 대해 Robust함
  - 대용량 데이터를 병렬처리하여 빠른 학습 가능

### "User-Free" Model-based Approaches
- cold start user 대처 가능
- Sequential scenario에 대처 가능. MF는 sequence를 고려하지 않음
- 해결 방안
  - Item vector를 입렬으로 ㅂ다아서 추천 결과를 생성하는 형태의 모델 생성
  - $f(u,i) = R_u \cdot W_i = \sum R_{u,j} W_{i,j}$ (이력이 존재하는 item에 대해서만,)
  - item간의 유사도(W)와 item에 대한 user rating(R)을 inner product 함으로써 compatibility function 계산
  - Item-based CF와 유사한 로직. 하지만 metric이 다름(Item-based : cosine, jaccard, pearson, euclidean... vs __parameter training__ )
- SLIM(Sparse Linear Methods for Top-N Recommender Systems)
  - 행렬 W를 학습하기 위해서 elastic net regularization term 추가
  - 위의 식에 따라 W를 학습하고 __새로운 사용자가 추가되더라도 학습된 W가 있으므로 추천 결과 생성 가능__
- FISM(Factored Item Similarity Model)
  - 

## Bayesian Personalized Ranking(BPR)
- 사용자의 선호도를 두 아이템간의 pairwise-ranking 문제로 formulation 함으로써 각 사용자의 personalized ranking function을 추정
- implicit feedback의 경우 사용자가 싫어하는 아이템과 아직 접하지 못한 아이템이 모두 '0'으로 취급되므로 __unseen item에 대해 negative scoring('0')하는 케이스 존재__
- Positive instance에는 높은 점수를 주고, non-positive의 경우에는 낮은 score를 주고 ranking하는 방식으로 변경
- __한 유저의 서로 다른 아이템에 대한 상대적 선호도를 비교하는 것__ -> __어떤 아이템에 대해서 다른 아이템보다 이를 선호하는 지를 예측하는 "classification"__


## Factorizing Personalized Markov Chains
- 사용자와 아이템 간의 관계 및 __아이템과 바로 이전 아이템의 관계__ 를 함께 모델링
- cf) Markov property : 다음 스텝의 상태는 이전의 상태에만 영향을 받음
- 유저와 다음 아이템의 compatibility + 이전 아이템과 다음 아이템의 compatibility + 사용자와 이전 아이템의 compatibility

## Personalized Ranking Metric Embedding(PRME)
- Compatibility를 계산하는 방식으로 inner product 대신, __Euclidean Distance__ 를 사용
