- Cross-validation
    - K-Fold Validation
- Bias vs Variance
    - 데이터가 한 점으로 모이게 된다 → Low Variance
    - 데이터가 평균적으로 Ground Truth로 모이게 된다 → Low Bias
    - Trade-off 존재
    - Cost = bias^2 + variance + noise
- Bootstrapping
    - 전체 데이터를 sampling을 통해서 여러가지 model을 만들어 확인하는 것
    - Bagging vs Boosting
        - Boosting : weak learners → strong learner
            - Sequential Model
            - e.g. Adaboost, Gradient boosting(XGBoost, lightGBM, Catboost...)
        - Bagging : n-models
            - N개의 샘플을 복원추출로 N번 추출
            - e.g. RandomForest
            - 각 샘플링에서 추출되지 않은 샘플은 model evaluation에 사용됨
- Batch size
    - Large batch size → Sharp Minimizer
    - **Small batch size → Flat Minimizer → 경험적으로 더 좋다!**
- Optimization
    - https://dev-jm.tistory.com/10
    - (stochastic) gradient descent
        - 일반적인 optimization method
        - 적절한 Learning rate 설정에 어려움이 있음
    - Momentum
        - $a_{t+1} \leftarrow \beta a_t + g_t$
        - accumulation ← momentum + new_gradient
        - 이전의 gradient를 유지시켜주는 성격이 있다
    - Nesterov Accelerated Gradient
        - Lookahead gradient
        - Momentum에서 관성의 성질로 인해 converge 하지 못하는 단점을 보완
    - Adagrad
        - 조금 변한 파라미터에 대해서는 lr를 낮게
        - 많이 변한 파라미터에 대해서는 lr를 높게
    - Adadelta
        - no learning rate in Adadelta
        - EMA(Exponential Moving Average)
    - RMSprop
    - Adam
        - Momentum + EMA
- Regularization
    - Early Stopping
        - 학습 수가 많아질수록 Loss가 커지기 전에 학습을 중단
    - Parameter norm penalty
    - Data Augmentation
        - 주어진 데이터를 활용해 데이터 셋의 크기를 늘리는 것
    - Noise Robustness
        - 일부러 noise를 넣어서 모델을 강화시키는 것
    - Label Smoothin
        - 서로 다른 label의 데이터를 섞어서 학습시키는 것
        - e.g. CutMix, Mixup, Cutout
    - Dropout
        - Neural Network의 weight를 일부러 죽이는 것
    - Batch-normalization
        - Mini Batch의 statistics를 normalize 시키는 것