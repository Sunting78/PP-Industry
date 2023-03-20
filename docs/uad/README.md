# 无监督异常检测工具

## 1. 环境依赖

* python
* paddlepaddle-gpu
* tqdm
* sklearn
* matplotlib
* pandas
* Pillow


## 2. 数据集

此工具以MVTec AD数据集为例, [下载MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/);


## 3. 训练, 评估, 预测命令

以PaDiM模型为例:

Train:

```python tools/uad/padim/train.py --config /ssd1/zhaoyantao/PP-Industry/configs/uad/padim/padim_resnet18_mvtec.yml --category bottle```

Eval:

```python tools/uad/padim/val.py --config /ssd1/zhaoyantao/PP-Industry/configs/uad/padim/padim_resnet18_mvtec.yml --category bottle```

Predict:

```python tools/uad/padim/predict.py --config /ssd1/zhaoyantao/PP-Industry/configs/uad/padim/padim_resnet18_mvtec.yml --category bottle```



## 4. 配置文件解读

无监督异常检测(uad)模型的参数可以通过YML配置文件和命令行参数两种方式指定, 如果YML文件与命令行同时指定一个参数, 命令行指定的优先级更高, YML文件主要包含以下参数:

```
# common arguments
device: gpu
seed: 3   # 指定numpy, paddle的随机种子

# dataset arguments
batch_size: 1
category: bottle # 指定MVTecAD数据集的某个类别
resize: [256, 256]  # 指定读取图像的resize尺寸
crop_size: [224, 224]  # 指定resize图像的crop尺寸
data_path: data/mvtec_anomaly_detection  # 指定MVTecAD数据集的根目录
save_path: output/    # 指定训练时模型参数保存的路径和评估/预测时结果图片的保存路径

# train arguments
do_eval: True  # 指定训练后是否进行评估
backbone: resnet18  # 支持resnet18, resnet50, wide_resnet50_2

# val and predict arguments
save_pic: True   # 指定是否保存第一张评估图片/预测图片的结果
model_path: output/resnet18/bottle/bottle.pdparams # 指定加载模型参数的路径

# predict arguments
img_path: data/mvtec_anomaly_detection/bottle/test/broken_large/000.png  # 指定预测的图片路径
threshold: 0.5    # 指定预测后二值化异常分数图的阈值
```

## 5. 集成模型

目前uad工具集成了[PaDiM](../../configs/uad/padim/README.md)模型, [PatchCore](../../configs/uad/patchcore/README.md)模型;