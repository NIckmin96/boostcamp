# RNN(Recurrent Neural Network)
- 은닉층(hidden layer)의 노드에서 활성화 함수를 통해 나온 결과값을 출력층으로 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 갖는다.
    - ![image](https://user-images.githubusercontent.com/81205952/203042419-7a84eba5-d723-4e94-a35d-5d36d4cab26c.png)
    - RNN의 cell(메모리 셀)은 이전의 값을 기억하려고 하는 메모리 역할을 수행
    - 현재 t시점에서 셀이 갖고 있는 값은 과거 메모리 셀의 값에 영향을 받은 값이고 현재(t)메모리 셀이 갖고 있는 값을 __hidden state__라고 한다.
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


- “the first sequence transduction model based entirely on **attention**”
- **sequence-to-sequence** [https://wikidocs.net/24996]
    - Encoder, Decoder 구조를 갖는다.
    - encoder, decoder는 각각 여러개의 RNN 셀로 구성이 되어있다.
    - <img src = "https://wikidocs.net/images/page/24996/%EC%9D%B8%EC%BD%94%EB%8D%94%EB%94%94%EC%BD%94%EB%8D%94%EB%AA%A8%EB%8D%B8.PNG">
    - Encoder에서 각각의 단어에 대해 embedding된 결과를 입력으로 받아 Encoder의 RNN Structure를 통과한 후, 마지막 cell의 hidden state를 Decoder의 첫번째 cell의 hidden state input으로 넣어주는데 이를 __context vector__라고 한다.
    - Decoder에서는 각 셀의 output gate에서 나온 결과에 softmax함수를 적용해 모델이 가지고 있는 단어 후보군으로부터 어떤 단어를 출력할지 계산하고 결정하게 된다.
    - <img src = "https://wikidocs.net/images/page/24996/decodernextwordprediction.PNG">
    - 하나의 모델에서입력 시퀀스와 출력 시퀀스의 개수가 다를 수 있다.
    - 입력 시퀀스의 도메인과 출력 시퀀스의 도메인이 다를 수 있다.
    - e.g. French → English
- Attention
    - seq2seq의 단점   
        - seq2seq은 하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하기 때문에 정보 손실 발생
        - RNN모델의 문제인 Vanishing Gradient   
    - __Decoder에서 출력 단어를 예측하는 매시점마다 인코더에서의 전체 입력 문장을 다시 한번 참고하게 된다.__   
    - Query, Key, Value
        - Attention(Q,K,V) = Attention Value
        - Attention함수는 주어진 Query에 대해서 모든 Key와의 유사도를 각각 구하고, 이 유사도를 Value에 반영해주고 그 Value vector를 모두 더해서 return
        - __Query__ : t 시점의 디코더 셀에서의 hidden state = decoder의 현시점(t)에서의 Hidden state
        - __Key__ : 모든 시점의 인코더 셀의 hidden state(mapping) = 
        - __Value__ : 모든 시점의 인코더 셀의 hidden state(mapping)
        - Key, Value 는 Dictionary에서 사용하는 개념과 같다.   
    - Dot product Attention
        - ![image](https://user-images.githubusercontent.com/81205952/194202905-b1050cbd-bd0f-4ce8-9ae9-0a4cd2ee5f94.png)

- Transformer의 기본 원리 [https://wikidocs.net/31379]
    - __IDEA__ : RNN구조를 따로 사용하지 않고, attention(encoder, decoder)만으로 모델을 구성
    - Encoder
        - Multi-head Self-attention + Feed forward Neural Network(FFNN)
        - Self-attention
            - Encoder내에서만 attention 수행
            - Query,Key,Value가 모두 Encoder에서 생성
            - input을 vector로 encoding할 때, 각각의 $x$만을 고려하는 것이 아니라, 다른 $x_i$들도 고려하게 된다.
            - Q,K,V를 얻는 방법
                - n개의 단어를 $d_{model}$의 차원 vector로 embedding 한 후에, $d_{model} * (d_{model}/num\ heads)$의 크기를 갖는 가중치 행렬과의 dot product연산으로 각각의 Q,K,V 벡터를 얻는다   
            - Embedding vector(Q,K,V) 생성 후,
                - __Scaled Dot product Attention 진행__
                1. Query , Key, Value vector를 embedding vector로 부터 만들어냄
                2. 그 후, query와 나머지 다른 단어들의 key vector의 내적으로 score를 계산
                3. 그 값을 sqrt(key vector’s dim)으로 나누고 softmax 함수에 대입 → # value vector의 weight
                4. 그 결과를 value 벡터에 곱해주고 sum을 취한 값을 사용 → **weighted sum of ‘value vector’**   
        - Transformer의 neural network는 가변적이고 유연한 모델이다 → 성능이 좋아짐
        - Multi headed attention(MHA) : Attention을 병렬적으로 수행
            - $d_{model}$의 차원을  $(d_{model}/num\ heads)$로 여러개로 나눠서 attention을 병렬적으로 진행후 concatenate
            - encoding 결과가 n개 나오게 된다
        - Positional encoding
            - self-attention 연산은 input의 순서를 고려하지 않는다
            - 따라서, 주어진 입력에 어떤 값을 더해준다
        - Self Attention 후에는 feed forward 연산을 수행한다
    - Decoder
        - key, value를 encoder에서 decoder로 보낸다 → Encoder-Decoder Attention
        - 일반 seq2seq과 다르게 transformer는 input 단어의 encoding 결과를 순차적이지않고 한번에 받는다 -> 미래 시점의 단어도 참고할 수 있는 현상 발생 -> __look-ahead mask__ 도입
        - Look-ahead Mask
            - 이전 단어들에 대해서만 dependent, 이후 단어들에 대해서는 independent하게 학습 : __Masked Decoder Attention__
        - output 결과는 순서대로 나오게 됨
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
