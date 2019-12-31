# 基于X光图片的充电宝检测
使用ssd网络结构来设计实现检测器，对X光目标图片中的有电芯与无电芯的充电宝进行检测定位。


<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- 安装`python3.7`及相关的开发依赖 `pip install -r requirements.txt`
- 克隆仓库
  * Note: 建议使用python3+.


## Datasets
数据集使用的是SIXray数据集，数据为带电池核心与无电池核心的充电宝。



## Training SSD
- 首先需要下载vgg16与与训练好的模型，可以从该链接下载：https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth； 在weight文件夹内有与训练的vgg16模型。
- 默认的模型保存在`weights/` 文件夹内。


```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- 训练模型预先指点好相关的**图片路径**与**标签路径**后执行`python train.py`

```Shell
python train.py --dataset_root path_to_datasets --epochs 50 
```

## Evaluation
网络评估验证

```Shell
python eval.py --trained_model path_to_model --six_root path_to_test_datasets 
```

## 模型测试
将测试文件的测试结果生成相应的文件保存在`predicted_file`中
```Shell
python -c "from test import *; test(img_path,anno_path)"
```
## Performance


##### mAP

| mAP | core_AP |coreless_AP |
|:-:|:-:|:-:|
|94.53 % | 94.69 % | 94.37% | 





## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thank you to [Alex Koltun](https://github.com/alexkoltun) and his team at [Webyclip](http://www.webyclip.com) for their help in finishing the data augmentation portion.
- A list of other great SSD ports that were sources of inspiration (especially the Chainer repo):
  * [Chainer](https://github.com/Hakuyume/chainer-ssd), [Keras](https://github.com/rykov8/ssd_keras), [MXNet](https://github.com/zhreshold/mxnet-ssd), [Tensorflow](https://github.com/balancap/SSD-Tensorflow)
