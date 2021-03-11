# Detecting-autism-with-Densenet
## 1、	模型

使用Densenet161，修改模型如下：

  修改第一层卷积为nn.Conv2d(3, 96, 7, 2, 3)

  添加最大池化层

  添加全连接层作为分类器
  
  数据集：https://www.kaggle.com/gpiosenka/autistic-children-data-set-traintestvalidate
  
  预训练权重：https://disk.pku.edu.cn:443/link/AF9235FA8640C0FEB8661A803FC65FE7  有效期限：2023-04-05 23:59
 

## 2、	训练

迭代训练60次，batchsize从56改为32，学习率0.0005不变

## 3、	结果

测试集精度82.7%

模型文件：https://disk.pku.edu.cn:443/link/55137C5409F04BD8D71B2F0F7E254E36 有效期限：2023-05-05 23:59

