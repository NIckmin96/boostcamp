 # 행렬의 개념
 - 행렬은 벡터공간에서 사용되는 연산자(operator)의 개념으로 이해된다
 - 행렬곱을 통해서 한 벡터를 다른 차원공간의 벡터로 변환할 수 있다 : $R^n \rightarrow R^m$
 # 역행렬
 - 여떤 행렬 $A$와 연산을 했을 때, 그 결과가 항등행렬(Identity Matrix) $I$로 나타나는 행렬
 - Determinant가 0이 아닌 경우에만 역행렬이 존재한다
 - 유사역행렬 / 무어 펜로즈 역행렬
   - $n>=m : A^+ = (A^TA)^{-1}A^T$
   - $n<=m : A^+ = A^T(AA^T)^{-1}$
 - 'np.linalg.pinv'를 이용하면 연립방정식의 해를 구할 수 있다
 # 선형회귀분석
 - ${\beta}$를 추정해야 하는 선형회귀식에서 역행렬 계산$(X{\beta} = y {\rightarrow} {\beta} = X^+y)$을 통해 ${\beta}$를 계산할 수 있다
 - sklearn의 LinearRegression과 같은 Logic
