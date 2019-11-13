# Person-ReID
全国人工智能大赛

1. Install dependencies:
    - [pytorch>=0.4](https://pytorch.org/)
    - torchvision
    - [ignite=0.1.2](https://github.com/pytorch/ignite) (Note: V0.2.0 may result in an error)
    - [yacs](https://github.com/rbgirshick/yacs)

2. Bag of tricks
- Warm up learning rate
- Random erasing augmentation
- Label smoothing
- Last stride
- BNNeck
- Center loss
- Focal loss
- OSM Loss(https://arxiv.org/pdf/1811.01459v2.pdf)
- GCNet(https://arxiv.org/abs/1904.11492?context=cs.LG)
