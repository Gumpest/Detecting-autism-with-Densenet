# Detecting-autism-with-Densenet
## 1、	模型

使用Densenet161，修改模型如下：

  修改第一层卷积为nn.Conv2d(3, 96, 7, 2, 3)

  添加最大池化层

  添加全连接层作为分类器

## 2、	训练

迭代训练60次，batchsize从56改为32，学习率0.0005不变

## 3、	结果

测试集精度82.7%

