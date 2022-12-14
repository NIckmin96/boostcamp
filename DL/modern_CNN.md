- ILSVRC
    - 이미지 데이터 활용 경진 대회
- AlexNet
    - 8-layers
    - Key ideas : 그 당시에 new Paradigm이었다.
        - ReLU 사용 → Vanishing Gradient Problem 해결(?)
        - GPU implementation
        - Data augmentation
        - Dropout
- VGGNet
    - 3 x 3 filter만 사용
        - 5x5 하나를 사용하는 것과 output field size는 동일하지만, parameter수가 적어지는 효과가 있다.
    - VGG16, VGG19
    - **Q. 2 x 2 는 왜 사용 안하는가?**
- GoogLeNet
    - 22 Layers
    - Network-in-Network(NIN)
    - Inception blocks
        - 1 x 1 convolution layer
        - parameter 수를 줄일 수 있음
        - **Q. 1x1 conv layer의 depth를 32와 다른 수로 하면 결과가 어떻게 되는가?**
- ResNet
    - Layer가 깊어짐에 따라, 어느 정도 깊이 이상의 neural network에서는 학습이 되지 않게 되었다
    - Identity Map(skip connection) 추가
        - 출력값에 입력값을 더해서 다시 activation function의 입력값으로 사용
        - Residual을 학습시키도록 하는 것이 목표
        - x, f(x)의 차원을 맞춰주어야 한다
        - **Batch-norm after Convolution**
        - Bottlenet Architecture → 원하는 dimension을 맞추기 위해 conv 앞뒤에 1x1 conv 추가
            - parameter 수 감소
- DenseNet
    - **Concatenation** instead of addition
    - Dense Block
        - concatenation
    - Transition Block
        - 1x1 conv → parameter수 감소
