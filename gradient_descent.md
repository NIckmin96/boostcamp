# Gradient Descent

## Gradient
- Gradient란? 
  - 간단히 말해, '미분값'을 의미한다
- Gradient를 계산해야하는 이유?
  - ML/DL에서 최종 목표는 실제값( $y$ )와 예측값( $\hat{y}$ )의 오차( $Error$ )를 최소화해 예측력을 높이는 것
  - 오차를 최소화하기 위해서 오차항의 미분값을 계산하여 미분값이 0에 근사할 때까지 Parameter를 업데이트 해주어야 한다.   
  - $$parameter = learing\ rate * grad$$

```
  var = init
  grad = gradient(var)
  # eps : 컴퓨터로 계산할 경우, 미분값이 정확히 0이 되는 것은 불가능하므로 매우 작은 값을 threshold로 사용한다.
  while(abs(grad) > eps) : 
    var = var - lr * grad
    grad = gradient(var)
   
```

### 변수가 다변량인경우 (Multi-Variate)
- 위의 방식과 동일
- 다만, 편미분을 이용(Partial differentiation)
- Gradient vector : 각 변수 별로 편미분을 계산한 값(각 변수별 편미분값)의 벡터
```
# Multi-Variate Case
var = init
grad = gradient(var)
# norm(grad) : gradient vector를 사용하기 때문에, norm을 계산해야한다.
while(norm(grad) > eps) : 
  var = var - lr*grad
  grad = gradient(var)
  
```
