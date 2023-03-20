# PatchCore
此模型是使用Paddle复现论文: [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/pdf/2106.08265v2.pdf).

## MVTec AD数据集上的实验结果

* 图像级和像素级ROCAUC指标:

|MvTec| ResNet18(Image-level) | ResNet18(Pixel-level) | ResNet18(PRO_score) |
|:---:|:---------------------:|:---------------------:|:-------------------:|
|Carpet|         0.992         |         0.991         |        0.961        | 
|Grid|         0.971         |         0.978         |        0.923        | 
|Leather|           1           |        0.9990         |        0.971        | 
|Tile|         0.990         |         0.937         |        0.848        | 
|Wood|         0.990         |         0.942         |        0.893        | 
|Bottle|           1           |         0.982         |        0.947        | 
|Cable|         0.979         |         0.985         |        0.943        | 
|Capsule|         0.982         |         0.989         |        0.937        | 
|Hazelnut|         0.999         |         0.988         |        0.935        | 
|Metal nut|         0.997         |         0.984         |        0.939        | 
|Pill|         0.939         |         0.976         |        0.928        | 
|Screw|         0.947         |         0.994         |        0.971        | 
|Toothbrush|         0.942         |         0.991         |        0.918        | 
|Transistor|         0.993         |         0.957         |        0.910        | 
|Zipper|         0.971         |         0.987         |        0.957        | 
|All classes|         0.979         |         0.978         |        0.932        | 
