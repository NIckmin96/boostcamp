# RNN(Recurrent Neural Network)
- 은닉층(hidden layer)의 노드에서 활성화 함수를 통해 나온 결과값을 출력층으로 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 갖는다.
    - ![image](https://user-images.githubusercontent.com/81205952/203042419-7a84eba5-d723-4e94-a35d-5d36d4cab26c.png)
    - RNN의 cell(메모리 셀)은 이전의 값을 기억하려고 하는 메모리 역할을 수행
    - 현재 t시점에서 셀이 갖고 있는 값은 과거 메모리 셀의 값에 영향을 받은 값이고 현재(t)메모리 셀이 갖고 있는 값을 __hidden state__ 라고 한다.
    - RNN은 'one-to-many', 'many-to-one', 'many-to-many'등의 형태로 설계 가능하고 각각의 형태에 따라 쓰임이 다르다.

## 간단한 수식 정의
- ![image](https://user-images.githubusercontent.com/81205952/203043343-b08101e3-2243-46d8-841f-642d0d4ffbdf.png)
- hidden layer : $h_t = tanh(W_x x_t + W_h h_{t-1} + b)$
- output layer : $y_t = f(W_y h_t + b)$
    - output을 sequence별로 각각 출력하게 되면 many-to-many 문제를 풀 수 있고
    - 1개의 output만 출력하게 되면 many-to-one 문제를 풀 수 있다.

## Deep RNN
- 다수의 은닉층을 갖는 RNN
- ![image](https://user-images.githubusercontent.com/81205952/203045422-372ff424-f6ba-4c93-97d5-9baf2271fec9.png)

## Bidirectional RNN
- t 시점에서의 출력값을 예측할 때 이전 시점 뿐만 아니라 이후 시점의 입력 또한 예측에 사용할 수 있다는 아이디어
- ![image](https://user-images.githubusercontent.com/81205952/203045522-7d31ed35-3ce7-4db7-a94c-018c114427b4.png)
- 하나의 출력값을 예측하기 위해 두개의 메모리 셀을 사용
    - Forward States(이전 시점의 hidden state를 받아 현재 은닉 상태를 계산)
    - Backward States(이후 시점의 hiddent state를 받아 현재의 은닉 상태를 계산)



# LSTM(Long Short-Term Memory)
## 기존 RNN(Vanila RNN)의 한계
- 비교적 짧은 sequence에 대해서만 효과를 보임
- sequence가 길어질 수록 앞의 정보가 사라지게 됨

## LSTM의 특징
![image](https://user-images.githubusercontent.com/81205952/203050704-7c8194c5-f0b8-4242-9666-51e8f2067568.png)
- Hidden layer의 메모리 셀에 __[입력 게이트(input gate), 망각 게이트(forget gate), 출력 게이트(output gate)]__ 를 추가
- 불필요한 기억을 지우고, 기억해야할 것들을 정함
- __Cell state__ : $C_t$
    - ![image](https://user-images.githubusercontent.com/81205952/203050810-b52e91f4-8da5-4a0f-a4ba-d292d94b4160.png)
    - Hidden state와 비슷하지만 3개의 게이트로 이루어져있다는 것이 차이점
    - 3개의 게이트에는 공통적으로 __시그모이드 함수__ 가 존재
        - 시그모이드 함수의 출력값(0~1)을 통해서 게이트를 조절하는데 사용
    - Input Gate(입력 게이트)
    - ![image](https://user-images.githubusercontent.com/81205952/203048813-5d61116f-e33e-402d-b511-940c4cb65a4c.png)
        - 현재 정보를 기억하기 위한 게이트
        - $i_t = \sigma(W_x x_t + W_{hi} h_{t-1} + b_i)$
            - 현재 t시점 input과 t-1시점의 hidden state의 Linear combination에 sigmoid 함수를 씌운 값
        - $g_t = tanh(W_{xg} x_t + W_{hg} h_{t-1} + b_g)$
            - 현재 t시점 input과 t-1시점의 hidden state의 Linear combination에 tanh 함수를 씌운 값
    - Forget Gate(삭제 게이트)
    - ![image](https://user-images.githubusercontent.com/81205952/203053859-4aee8894-1e78-46d4-bac9-a252006f5e8e.png)
        - 기억을 삭제하기 위한 게이트
        - $f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)$
            - 현재 t시점 input과 t-1시점의 hidden state의 Linear combination에 sigmoid 함수를 씌운 값
            - 이 값이 삭제 과정을 거친 정보의 양
            - __0에 가까울 수록 많이 삭제된 것, 1에 가까울 수록 온전히 기억한 것__
    - Cell state
        - ![image](https://user-images.githubusercontent.com/81205952/203054503-ae0d7fd9-1995-4260-bc9f-29d3396f23f8.png)
        - $C_t = f_t \circ C_{t-1} + i_t \circ g_t$
        - input gate에서 구한 값( $g_t, i_t$ )에 대해 elementwise-product를 계산( $\circ$ )
        - input gate에서 선택된 기억을 forget gate의 결과값과 더함 -> __Cell state__
    - Output gate
        - $o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)$
        - $h_t = o_t \circ tanh(c_t)$
        - 현재 시점 t의 x값과 이전 시점 t-1의 은닉 상태(hidden state)가 시그모이드 함수를 지난 값
        - 현재 시점 t의 hidden state를 결정하는 일에 사용

# GRU(Gated Recurrent Unit)

- LSTM의 장기 의존성에 대한 해결책을 유지하면서, hidden state를 업데이트하는 연산량을 줄임(__LSTM 간소화__)
- __Update Gate, Reset Gate__ 두가지 게이트만 존재
- 경험적으로, 데이터 양이 적을 때에는 GRU가 조금 더 낫고, 많으면 LSTM이 낫다고 함

# Encoder-Decoder

- 두개의 RNN을 사용해서, 하나는 Encoder, 다른 하나는 Decoder로 명명하고 두개의 RNN을 연결해서 사용하는 구조
- 주로 입력 sequence와 출력 sequence가 달을 경우에 사용
    - 번역기, 텍스트 요약 등

# Sequence-to-Sequence(Seq2Seq)

- Encoder, Decoder 구조를 갖는다.
- encoder, decoder는 각각 여러개의 RNN 셀로 구성이 되어있다.
- <img src = "https://wikidocs.net/images/page/24996/%EC%9D%B8%EC%BD%94%EB%8D%94%EB%94%94%EC%BD%94%EB%8D%94%EB%AA%A8%EB%8D%B8.PNG">
- Model Architecture
    - Encoder에서 각각의 단어에 대해 embedding된 결과를 입력으로 받아 Encoder의 RNN Structure를 통과한 후, 마지막 cell의 hidden state를 Decoder의 첫번째 cell의 hidden state input으로 넣어주는데 이를 __context vector(CV)__ 라고 한다.
    - Decoder에서는 각 셀의 output gate에서 나온 결과에 softmax함수를 적용해 모델이 가지고 있는 단어 후보군으로부터 어떤 단어를 출력할지 계산하고 결정하게 된다.
    - Decoder는 초기 입력값으로 문장의 시작을 의미하는 __<sos>__ 가 들어가고 문장의 끝을 의미하는 __<eos>__ 가 예측될 때까지 output의 예측을 반복한다.
    - <img src = "https://wikidocs.net/images/page/24996/decodernextwordprediction.PNG">
- 하나의 모델에서입력 시퀀스와 출력 시퀀스의 개수가 다를 수 있다.
- 입력 시퀀스의 도메인과 출력 시퀀스의 도메인이 다를 수 있다.
- e.g. French → English
    
# Attention

- seq2seq의 단점   
    - seq2seq은 하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하기 때문에 정보 손실 발생
    - RNN모델의 문제인 Vanishing Gradient   
- Attention의 특징
    - __Decoder에서 출력 단어를 예측하는 매시점마다 인코더에서의 전체 입력 문장을 다시 한번 참고하게 된다__    
- Query, Key, Value
    - Attention(Q,K,V) = Attention Value
    - Attention함수는 주어진 Query에 대해서 모든 Key와의 유사도를 각각 구하고, 이 유사도를 Value에 반영해주고 그 Value vector를 모두 더해서 return
    - __Query__ : t 시점의 디코더 셀에서의 hidden state = decoder의 현시점(t)에서의 Hidden state
    - __Key__ : 모든 시점의 인코더 셀의 hidden state(mapping)
    - __Value__ : 모든 시점의 인코더 셀의 hidden state(mapping) -> __Query, Key 벡터간의 유사도를 반영한 모든 시점에서의 hidden state__
    - Key, Value 는 Dictionary에서 사용하는 개념과 같다.   
- Dot product Attention
    - ![image](https://user-images.githubusercontent.com/81205952/194202905-b1050cbd-bd0f-4ce8-9ae9-0a4cd2ee5f94.png)
    - 디코더의 세번째 셀에서 출력 단어를 예측할 떄를 가정
    - 출력 단어를 예측하기 위해 인코더의 모든 단어들의 embedding vector를 참고
    - 위의 빨간 bar plot을 통해 softmax함수의 연산 결과를 표시, 직사각형의 크기가 클수록 도움이 되는 정도의 크기가 큼
    - 이 정보를 하나의 정보(초록색 화살표)로 담아서 디코더로 전송   
- Dot Product Attention 연산 과정
    1. Attention score 구하기
        - Attention Score : __현재 디코더의 시점 t에서 단어를 예측하기 위해, 인코더의 모든 은닉 상태 각각이 디코더의 현 시점의 은닉 상태( $s_t$ )와 얼마나 유사한지를 판단하는 score__
        - ![image](https://user-images.githubusercontent.com/81205952/203197262-9f78a1b2-c29c-4eda-8344-34b2f270cb9c.png)
        - $s_t^T \cdot h_i$ : 현 시점의 은닉 상태( $s_t$ )와 인코더의 각 시점의 은닉상태( $h_i$ )에 대한 내적
        - ![image](https://user-images.githubusercontent.com/81205952/203198266-57df5714-0a10-46ab-b3d9-f69a68e73914.png)   
        - 계산된 각 attention score를 모아 $e^t$ 라고 정의
        - $e^t = [s_t^T h_1,...,s_t^T h_N]$
    2. Softmax를 통해 Attention Distribution을 구하기
        - $e^t$에 소프트맥스 함수를 적용 : __Attention Distribution__
        - ![image](https://user-images.githubusercontent.com/81205952/203199079-eb139c37-0bbb-42b7-9b87-cfff422b5338.png)
        - Attention distribution의 각각의 값 : __Attention Weight__ -> 위 그림의 빨간 bar plot
        - $\alpha^t = softmax(e^t)$
    3. 각 인코더의 attention weight와 hidden state를 가중합하여 attention value를 구한다
        - ![image](https://user-images.githubusercontent.com/81205952/203199290-99ce64d1-b030-42a8-9d46-dacefe0bf0f7.png)
        - $a^t = \sum_{i=1}^{N} \alpha_i^t h_i$
        - attention값은 종종 context vector로 불리기도 함
    4. 어텐션 값과 디코더의 t 시점의 은닉 상태를 연결(concatenate)
        - ![image](https://user-images.githubusercontent.com/81205952/203199554-1ee3d08c-e80f-4b0d-ade3-bbd626c98043.png)
        - $a_t$(어텐션 값)와 $s_t$(디코더의 t시점의 hidden state)를 concat = $v_t$
        - $v_t$를 디코더의 출력 예측값의 입력으로 사용 -> 인코더로부터의 정보를 좀 더 잘 활용할 수 있게 됨
    5. 출력층 연산의 입력이 되는 $\tilde{s_t}$ 를 계산
        - ![image](https://user-images.githubusercontent.com/81205952/203200497-09cb67f1-28ed-4213-8f40-4bcfbb3d2172.png)
    6. $\tilde{s_t}$ 를 출력층의 입력으로 사용
        - $\hat{y_t} = Softmax(W_y \tilde{s_t} + b_y)$
    
    
# Transformer

- “the first sequence transduction model based entirely on **attention**”
- Transformer의 기본 원리 [https://wikidocs.net/31379]
    - __IDEA__ : RNN구조를 따로 사용하지 않고, attention(encoder, decoder)만으로 모델을 구성
    - Encoder
        - __Multi-head Self-attention + Feed forward Neural Network(FFNN)__
        - __Self-attention__
            - __Encoder내에서만 attention 수행__
            - Query,Key,Value가 모두 Encoder에서 생성
            - input을 vector로 encoding할 때, 각각의 $x$만을 고려하는 것이 아니라, 다른 $x_i$들도 고려하게 된다.
            - Self Attention 과정
                1. Query , Key, Value vector를 embedding vector로 부터 만들어냄
                - ![image](https://user-images.githubusercontent.com/81205952/203205055-a276b6ce-e1d2-4a32-ad45-9ee1186dc63c.png)
                - Q,K,V를 얻는 방법
                    - n개의 단어를 $d_{model}$의 차원 vector로 embedding 한 후에, $d_{model} * (d_{model}/num\ heads)$의 크기를 갖는 가중치 행렬과의 dot product연산으로 각각의 Q,K,V 벡터를 얻는다   
                - ![image](https://user-images.githubusercontent.com/81205952/203206308-454a0c36-30cb-4f02-a537-ce41b0d37cf5.png)
                2. 그 후, query와 나머지 다른 단어들의 key vector의 내적으로 score를 계산 : __Scaled Dot Product__
                3. 그 값을 sqrt(key vector’s dim)으로 나누고 softmax 함수에 대입 → __value vector의 weight__ : __Scaled Dot Product__
                - ![image](https://user-images.githubusercontent.com/81205952/203206415-61a33641-0b0d-4fa4-93c8-c999c804b165.png)
                4. 그 결과를 value 벡터에 곱해주고 sum을 취한 값을 사용 → **weighted sum of ‘value vector’**   
        - __Multi headed attention(MHA) : Attention을 병렬적으로 수행__
            - ![image](https://user-images.githubusercontent.com/81205952/203206918-166c9024-7d27-4f78-99c0-02cfa1cbd19b.png)
            - $d_{model}$의 차원을  $(d_{model}/num\ heads)$로 여러개로 나눠서 attention을 병렬적으로 진행후 concatenate
            - encoding 결과가 n개 나오게 된다
            - ![image](https://user-images.githubusercontent.com/81205952/203207176-9c363b6b-d534-4b03-9596-217d346fb3de.png)
            - __각 head의 attention 결과를 concatenate한 후, 새로운 가중치 행렬에 대해 연산한 값을 Multi-head Attention값으로 최종적으로 사용__
        - Positional encoding
            - self-attention 연산은 input의 순서를 고려하지 않는다
            - 따라서, 주어진 입력에 어떤 값을 더해준다
        - Self Attention 후에는 feed forward 연산을 수행한다   
    - Decoder
        - 첫번째 층 : __self attention & look-ahead mask__
            - 일반 seq2seq과 다르게 transformer는 input 단어의 encoding 결과를 순차적이지않고 한번에 받는다 -> 미래 시점의 단어도 참고할 수 있는 현상 발생
            - __look-ahead mask__ 도입
            - Look-ahead Mask
                - 이전 단어들에 대해서만 dependent, 이후 단어들에 대해서는 independent하게 학습 : __Masked Decoder Attention__
        - 두번째 층 : __인코더-디코더 어텐션__
            - key, value를 encoder에서 decoder로 보낸다 → Encoder-Decoder Attention
            - self attention이 아니다
        - output 결과는 순서대로 나오게 됨   
    - Padding Mask
        - ![image](https://user-images.githubusercontent.com/81205952/203206980-5d460c2b-edb3-4f25-b26e-aa9152c87f64.png)
        - 패딩이 있을 경우 어텐션 연산에서 제외하기 위해 씌워주는 것
    - Position-wise FFNN
        - FFNN은 인코더와 디코더에서 공통적으로 가지고 있는 서브층
- 다른 분야에 적용된 Transformer 기법
    - Vision Transformer
        - 이미지 데이터에 대해서 transformer 사용
    - DALL-E
        - 문장이 주어졌을 때, 이미지를 생성
- Embedding과 Encoding의 차이
    - [https://jamesmccaffrey.wordpress.com/2022/08/02/the-difference-between-encoding-embedding-and-latent-representation-in-my-world/]
    - Embedding
        - Embedding converts an integer word ID to a vector
        - e.g. “the” = 4  = [-0.1234, 1.9876,…, 3.4681]
    - Encoding
        - Encoding converts categorical data to numeric data
        - e.g. “red” = [ 0 1 0 0]
