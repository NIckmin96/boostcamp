# Intro

- Query : user가 원하는 정보의 keyword
- 추천 시스템의 필요성
    - 과거에는 유저가 접할 수 있는 상품, 컨텐츠가 제한적
    - 웹,모바일 환경은 다양한 상품, 컨텐츠를 등장하게 함
        - Long Tail Phenomenon
    - 정보를 찾는데 시간이 오래 걸림
        - 유저가 원하는 것을 어떤 키워드로 찾아야 하는지 모를 수 있음
- Long-Tail Recommendation 사례
    - 유튜브 동영상 추천
    - SNS 친구 추천
- 사용 데이터
    - user info
        - **User profiling** : user에 대한 정보를 구축
        - 식별자(Identifier)
            - 유저ID, 브라우저 ID, 쿠키
        - 데모그래픽 정보
            - 성별, 연령, 지역, 관심사 → 데이터를 구하기 어려울 경우, 추정도 함
        - 유저 행동 정보
            - 페이지 방문 기록, 아이템 평가, 구매 등의 피드백 기록
    - item info
        - Item Profiling : Item에 대한 정보를 구축
        - Item ID
        - Item Meta Data(아이템의 고유정보)
    - user-item interaction info
        - Explicit Feedback
            - 유저에게 아이템에 대한 피드백을 **직접** 물어본 경우
        - Implicit Feedback
            - 유저가 아이템을 클릭하거나 구매한 경우 → **추정**
- 추천 시스템의 목적
    - 특정 유저에게 적합한 아이템을 추천 or 특정 아이템에게 적합한 유저를 추천
    - 유저 - 아이템의 상호작용에 대해 평가할 수 있는 metric 필요
    - 추천을 위한 **score 계산!**
        - 랭킹 : 유저에게 적합한 아이템 Top K개를 추천하는 문제
            - 유저가 아이템에 가지는 정확한 선호도를 구할 필요는 없음 → **상대적 비교만 가능하면 됨**
            - Metrics
                - Precision@K
                - Recall@K
                - MAP@K
                - nDCG@K
        - 예측 : 유저가 아이템에 가질 정확한 선호도를 예측(평점, 구매, 클릭할 확률)
            - Explicit Feedback : 철수가 ‘아이언맨’에 매길 평점을 예측
            - Implicit Feedback : 영희가 아이폰12를 구매하거나 조회할 확률을 예측
            - **유저-아이템 행렬**을 채우는 문제
            - Metrics
                - MAE
                - RMSE
                - AUC
# 추천 시스템의 평가 
- 비즈니스 / 서비스 관점
    - 추천 시스템 적용으로 인해 매출, PV의 증가
        - **PV(Page Views) : 방문 횟수**
    - 추천 아이템으로 인해 유저의 CTR의 상승
        - **CTR(Click-Through Rate) : 노출 대비 클릭 수**
            - clicks / impressions * 100(%)
        - CVR(Conversion Rate) : 얼마나 많은 사람들이 광고를 본 후 행동을 전환했는가?
            - Converted / Clicked * 100(%)
- 품질 관점
    - 연관성(Relevance)
    - 다양성(Diversity)
    - 새로움(Novelty)
    - 참신함(Serendipity)
## Offline Test
- Offline Test
    - 새로운 추천 모델을 검증하기 위해 가장 우선적으로 수행되는 단계
    - Serving bias
        - Online Test에서는 user의 기록이 새롭게 data에 추가되어 model에 반영되므로 serving bias가 생긴다.
    - Precision@K
        - 우리가 추천한 K개 아이템 중 실제 유저가 관심있는 아이템의 비율
    - Recall@K
        - 유저가 관심있는 전체 아이템 중 우리가 추천한 아이템의 비율
    - Mean Average Precision(MAP)@K
        - [https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#The-"Mean"-in-MAP](https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#The-%22Mean%22-in-MAP)
        - AP@K
            - Precision@1 부터 Precision@K 까지의 평균값
            - 관련 아이템을 더 높은 순위에 추천할수록 점수가 상승함
            - Relevant한 항목에 대해서만 계산함
            - Example


                | Recommendations | Precision@K | AP@K |
                | --- | --- | --- |
                | [1,0,0] | [1/1, 1/2, 1/3] | 1/3*(1) = 0.33 |
                | [0,1,0] | [0, 1/2, 1/3] | 1/3*(1/2) = 0.15 |
                | [0,0,1] | [0, 0, 1/3] | 1/3*(1/3) = 0.11 |
        - MAP@K
            - 모든 유저에 대한 Average Precision 값의 평균
    - Normalized Discounted Cumulative Gain(NDCG)
        - Top K리스트를 만들고 유저가 선호하는 아이템을 비교하여 값을 구함
        - 추천의 순서에 가중치를 더 많이 둠
        - 값이 1에 가까울 수록 좋음
        - MAP와 달리 연관성을 binary값(yes/no)이 아닌 수치로도 사용가능, **how much**도 유추 가능
        - Cumulative Gain
            - 상위 K개 아이템에 대하여 관련도를 합한 것
            - 순서에 상관 없음
            - $CG_K = \Sigma_{i=1}^{k} rel_i$
        - Discounted Cumulative Gain
            - 순서에 따라 Cumulative Gain을 Discount함
            - $DCG_K = \Sigma_{i=1}^{k} \frac{rel_i}{log_2(i+1)}$
        - Ideal Gain
            - 이상적인 추천이 일어났을 때의 DCG값
            - 가능한 DCG값들 중에 가장 크다
            - $IDCG_K = \Sigma_{i=1}^{k} \frac{rel_i^{opt}}{log_2(i+1)}$
        - Normalized Gain
            - 추천 결과에 따라 구해진 DCG를 IDCG로 나눈 값
            - $NDCG = \frac{IDCG}{DCG}$
## Online Test
- Online Test
    - A/B Test
        - Offline Test에서 검증된 가설이나 모델을 이용해 실제 추천 결과를 서빙하는 단계
        - 동시에 평가
- 인기도 기반 추천
    1. 조회수가 가장 많은 아이템을 추천(Most Popular)
        - 뉴스 추천
        - Score Formula : $f(popularity, age)$ *age : 뉴스의 생성 날짜
            - 조회수가 빠르게 늘어나게 되면 시간이 지나도 예전 뉴스가 상위에 Rank되는 문제 존재
        - Hacker News Formula
            - gravity(=1.8)
            - $score = \frac{pageviews - 1}{(age+2)^{gravity}}$
            - 시간이 지날수록 age가 점점 증가하므로 score는 작아짐
        - Reddit Formula
            - $score = log_{10}(ups-donws) + \frac{sign(ups-downs) \cdot seconds}{45000}$
            - 최근 포스팅된 글에 더 높은 점수(가중치)를 부여
            - **seconds** : 글이 포스팅 된 절대 시간값
            - **log** : 초반 vote에 대해서는 높은 값, 나중의 vote에 대해서는 낮은 값 부여
    2. 평균 평점이 가장 높은 아이템을 추천(Highly Rated)
        - 맛집 추천
        - Score Formula : f(rating, # of ratings)
        - Steam Rating Formula
            - $avg\ rating = \frac{num\ of\ positive \ reviews}{num\ of\ reviews}$
            - $score = avg\ rating - (avg\ rating - 0.5) \cdot 2^{-log(num\ of\ reviews)}$
        - review의 개수가 아주 많을 경우 score는 평균 rating과 거의 유사해짐
            - 음수 항이 0에 가까워 지기 때문에
