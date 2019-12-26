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
- [OSM Loss](https://arxiv.org/pdf/1811.01459v2.pdf)
- [GCNet](https://arxiv.org/abs/1904.11492?context=cs.LG)

3. 初赛[pretrained-weights](https://pan.baidu.com/s/1EhC6doJvJH6uOX9fOe4pWw)
 提取码: kmn2 
 copy to folder 'market1501' 

4.训练方法
  1.将数据集train 与 train_list.txt放置在dataset文件夹下，运行change_name.py，预处理数据集，将query 解压在 query文件夹下，gallery 解压在 bounding_box_test文件夹下。    
  2.运行train_resnet152.sh 训练模型 模型结果及模型权重保存在market1501下。    
  
