# Gradient Descent

## Gradient
- Gradient란? 
  - 간단히 말해, '미분값'을 의미한다
- Gradient를 계산해야하는 이유?
  - ML/DL에서 최종 목표는 실제값( $y$ )와 예측값( $\hat{y}$ )의 오차( $Error$ )를 최소화해 예측력을 높이는 것
  - 오차를 최소화하기 위해서 오차항의 미분값을 계산하여 미분값이 0에 근사할 때까지 Parameter를 업데이트 해주어야 한다.   
  - $$parameter = learing\ rate * parameter$$

```
function test() {
  console.log("notice the blank line before this function?");
}


```


  function()  {
  var = init;
  grad = gradient(var);
  while(abs(grad) > eps) : 
    var = var - lr * grad;
    grad = gradient(var);
  }


