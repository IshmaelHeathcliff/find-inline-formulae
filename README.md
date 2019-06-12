# 论文图片中定位行内公式

## 数据处理

image_utils.py  图片处理工具
texf_topng.py tex文件转png图片
ttp.py ttpb.py 使用texf_topng.py的命令行实现
image_process.py 将png图片处理后生成tfrecords

## 网络训练

train.py 网络训练主体
inference.py 网络结构定义

## 训练结果使用

check.py 评估网络训练结果
formula_find.py 实际将论文图片经网络结果处理后框出公式位置
result_record 一些网络训练结果评估的记录

## 实际效果

models 网络模型及实际图片处理效果
