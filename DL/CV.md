- Semantic Segmentation : 어떤 이미지를 각 픽셀마다 분류하는 것
    - **Fully convolutional network**
        - Dense layer를 없애는 것
        - **parameter수는 같음**
        - “convolutionalization”
        - Independent to input image size
    - Deconvolution(conv transpose)
        - convolution의 역연산(so-called)
        - 줄어든 output에 padding을 많이 줘서 원래의 size를 갖는 output을 새로 생성하는 것
- Detection : 픽셀별로 분류하는 것이 아니라 boundary box를 만드는 것
    - R-CNN
        - 각 bounding box마다 CNN을 돌려야 하기 때문에 연산량이 많아진다
    - SPPNet
        - CNN 한번만 돌리기
        - CNN을 한번 거친 feature map 위에서 각 bounding box의 위치에 해당하는 sub-tensor를 불러와서 활용
    - Fast R-CNN
        - ROI pooling
    - Faster R-CNN
        - Region Proposal Network + Fast R-CNN
        - Region Proposal Network
            - Anchor box
            - 9*(4+2)
    - YOLO
        - it simultaneously predicts multiple bounding boxes and class probabilities