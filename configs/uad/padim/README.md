# PaDiM-Anomaly-Detection-Localization
此模型是使用Paddle复现论文: [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/pdf/2011.08785).

## MVTec AD数据集上的实验结果

* 图像级和像素级ROCAUC指标:

|MvTec| R18-Rd100(Image-level) | R18-Rd100(Pixel-level) |
|:---:|:----------------------:|:----------------------:|
|Carpet|         0.996          |         0.990          | 
|Grid|         0.928          |         0.944          | 
|Leather|           1            |         0.987          |
|Tile|         0.965          |         0.894          | 
|Wood|         0.981          |         0.931          |
|Bottle|         0.999          |         0.979          |
|Cable|         0.876          |         0.949          | 
|Capsule|         0.875          |         0.980          | 
|Hazelnut|         0.760          |         0.970          | 
|Metal nut|         0.984          |         0.963          |
|Pill|         0.844          |         0.923          |
|Screw|         0.774          |         0.975          |
|Toothbrush|         0.969          |         0.985          |
|Transistor|         0.972          |         0.974          |
|Zipper|         0.850          |         0.981          
|All classes|         0.918          |         0.962          | 
