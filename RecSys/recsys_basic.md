# 추천 시스템 기법

## 연관 분석

- 연관 규칙 분석
  - "장바구니 분석", "서열 분석"이라고도 불림
  - 연속된 거래들 사이의 규칙을 발견하기 위해 적용
  - e.g. 맥주와 기저귀를 같이 구매하는 빈도가 얼마나 되는가?
  - 빈발 집합(Frequent Itemset)
    - itemset
      - 1개 이상의 item의 집합
    - Support
      - itemset이 전체 transcation data에서 등장하는 비율
    - Support count
      - itemset이 전체 transcation data에서 등장하는 횟수(count)
    - Frequent itemset
      - 유저가 지정한 minimum support 값 이상의 itemset을 의미
      - 일정 비율 이상으로 등장하는 itemset을 의미
  - 연관 규칙 척도
    - Support
      - itemset X,Y를 모두 포함하는 transaction의 비율, 전체 transaction에 대한 itemset의 확률값
    - Confidence
      - X가 포함된 transaction 가운데, Y도 포함하는 transaction의 비율, Y의 X에 대한 조건부 확률
    - Lift
      - X가 포함된 transaction 가운데 Y가 등장할 확률 / Y가 등장할 확률
  - 연관 규칙의 탐색
    - 주어진 transaction 가운데, 의미 있는 연관 규칙만을 찾는다.
    - Brute Force
      - 가능한 모든 연관 규칙에 대해 metric을 계산하고 따져봄
      - 많은 연산량이 요구됨
    - Strategy
      - 가능한 후보 itemset의 개수를 줄인다
      - 탐색하는 transaction의 숫자를 줄인다
      - 탐색 횟수를 줄인다

## TF-IDF
- Item의 feature를 TF-IDF를 통해 하나의 벡터로 만들 수 있음 -> "Item Profile"
- User가 소비한 Item의 리스트(Item Profile의 List)를 합침 -> "User Profile"
- 유사도 
