# convert_tools使用说明

## 1 工具说明

convert_tools是PaddleIndustry提供的用于处理标注文件格式转换的工具集，位于`tools/convert_tools/`目录。由于工业质检可能采用检测，分割或者检测结合RoI分割的解决方案, 为了便于进行PPL的对比和尝试，进行一些简单的任务输入格式的文件处理工作。

*请注意，convert_tools目前为一项实验性功能，若您在使用过程中遇到问题，请及时向我们反馈。*

## 2 文件说明

目前convert_tools共有3个文件，各文件及其功能如下：
 
- `convert_mask_to_roi.py`:       全图分割转换为RoI区域分割；
- `convert_mask_to_coco.py`:      转换分割数据为coco格式的json文件，便于训练检测/实例分割；
- `convert_coco_to_roi_mask.py`:  将coco格式的json文件转化为RoI区域分割。


## 3 使用示例

### 3.1 示例数据集

本文档以COCO 2017数据集作为示例数据进行演示。您可以在以下链接下载该数据集：

- [官方下载链接](https://cocodataset.org/#download)
- [aistudio备份链接](https://aistudio.baidu.com/aistudio/datasetdetail/7122)

下载完成后，为方便后续使用，您可以将`coco_tools`目录从PaddleRS项目中复制或链接到数据集目录中。完整的数据集目录结构如下：

```
./COCO2017/      # 数据集根目录
|--train2017     # 训练集原图目录
|  |--...
|  |--...
|--val2017       # 验证集原图目录
|  |--...
|  |--...
|--test2017      # 测试集原图目录
|  |--...
|  |--...
|
|--annotations   # 标注文件目录
|  |--...
|  |--...
|
|--coco_tools    # coco_tools代码目录
|  |--...
|  |--...
```

### 3.2 全图分割标签转RoI分割

使用`convert_mask_to_roi.py`可以将全图分割转化为RoI分割，方便检测+RoI分割的质检PPL中，RoI分割模块数据的转换，输出图像，标签和txt文件。

#### 3.2.1 命令演示

执行如下命令，完成转化，RoI图像，标签以及RoI.txt保存在output路径中：

```
python3 tools/convert_tools/convert_mask_to_roi.py \
        --image_path ./dataset/kolektor2/images/val \
        --anno_path ./dataset/kolektor2/anno/val \
        --class_num 1 \
        --output_path output \
        --suffix _GT
        --to_binary
```

#### 3.2.2 参数说明


| 参数名          | 含义                                 | 默认值     |
| -------------  | ------------------------------------| --------- |
| `--image_path` |  需要转化的原图路径                    |           |
| `--anno_path`  |  原图对应的分割标签路径                 |           |
| `--class_num`  |  分割类别数                           |           |
| `--to_binary`  |  是否转化为2类训练                     | `False`   |
| `--suffix`     |  分割标签相对于图像的后缀               |  `" "`    |
| `--pad_scale`  |  剪切图像pad尺度                      |  `0.5`    |
| `--output_path`|  保存路径                             |`./output/`|



#### 3.2.3 结果说明

在`output/images/` 路径下保存RoI图像，注：输出的图像名为原始图像名结合类别id和RoI id;
在`output/annos/` 路径下保存RoI对应的分割标签，0是背景，1是前景；
`output/RoI.txt` 图像和标签列表，每一行表示对应的RoI图像和标签图像路径。

### 3.3 分割标签转coco格式json文件

使用`convert_mask_to_coco.py`可以让图像分割任务数据格式转化为coco json文件，方便用于检测任务。

#### 3.3.1 命令演示

执行如下命令，完成图像分割到coco格式json文件的转换：

```
python3 tools/convert_tools/convert_mask_to_coco.py \
        --image_path ./dataset/kolektor2/images/val \
        --anno_path ./dataset/kolektor2/anno/val \
        --class_num 1 \
        --output_name output \
        --suffix _GT
```

#### 3.3.2 参数说明

| 参数名          | 含义                                 | 默认值     |
| -------------  | ------------------------------------| --------- |
| `--image_path` |  需要转化的原图路径                    |           |
| `--anno_path`  |  原图对应的分割标签路径                 |           |
| `--class_num`  |  分割类别数                           |           |
| `--suffix`     |  分割标签相对于图像的后缀               |  `" "`    |
| `--output_name`|  保存json的文件名                     |  `coco`   |


#### 3.2.3 结果说明

生成`coco.json`， 包含`images`，`annotations`,`categories`字段。

### 3.4 统计目标检测标注框信息

使用`json_AnnoSta.py`，可以从`instances_val2017.json`中快速提取标注信息，生成csv表格，并生成统计图。

#### 3.4.1 命令演示

执行如下命令，打印`instances_val2017.json`信息：

```
python ./coco_tools/json_AnnoSta.py \
    --json_path=./annotations/instances_val2017.json \
    --csv_path=./anno_sta/annos.csv \
    --png_shape_path=./anno_sta/annos_shape.png \
    --png_shapeRate_path=./anno_sta/annos_shapeRate.png \
    --png_pos_path=./anno_sta/annos_pos.png \
    --png_posEnd_path=./anno_sta/annos_posEnd.png \
    --png_cat_path=./anno_sta/annos_cat.png \
    --png_objNum_path=./anno_sta/annos_objNum.png \
    --get_relative=True
```

#### 3.4.2 参数说明

| 参数名                  | 含义                                                                                                                       | 默认值         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `--json_path`          | （可选）需要统计的json文件路径                                                                                                 |               |
| `--csv_path`           | （可选）统计表格保存路径                                                                                                       | `None`        |
| `--png_shape_path`     | （可选）png图片保存路径，图片内容为所有目标检测框shape的二维分布                                                                    | `None`        |
| `--png_shapeRate_path` | （可选）png图片保存路径，图片内容为所有目标检测框shape比例(宽/高)的一维分布                                                           | `None`        |
| `--png_pos_path`       | （可选）png图片保存路径，图片内容为所有目标检测框左上角坐标的二维分布                                                                 | `None`        |
| `--png_posEnd_path`    | （可选）png图片保存路径，图片内容为所有目标检测框右下角坐标的二维分布                                                                 | `None`        |
| `--png_cat_path`       | （可选）png图片保存路径，图片内容为各个类别的对象数量分布                                                                           | `None`        |
| `--png_objNum_path`    | （可选）png图片保存路径，图片内容为单个图像中含有标注对象的数量分布                                                                   | `None`        |
| `--get_relative`       | （可选）是否生成图像目标检测框shape、目标检测框左上角坐标、右下角坐标的相对比例值<br />(横轴坐标/图片长，纵轴坐标/图片宽)                    | `None`        |
| `--image_keyname`      | （可选）json文件中，图像所对应的key                                                                                             | `'images'`    |
| `--anno_keyname`       | （可选）json文件中，标注所对应的key                                                                                             | `'annotations'`|
| `--Args_show`          | （可选）是否打印输入参数信息                                                                                                    | `True`        |

#### 3.4.3 结果展示

执行上述命令后，输出结果如下：

```
------------------------------------------------Args------------------------------------------------
json_path = ./annotations/instances_val2017.json
csv_path = ./anno_sta/annos.csv
png_shape_path = ./anno_sta/annos_shape.png
png_shapeRate_path = ./anno_sta/annos_shapeRate.png
png_pos_path = ./anno_sta/annos_pos.png
png_posEnd_path = ./anno_sta/annos_posEnd.png
png_cat_path = ./anno_sta/annos_cat.png
png_objNum_path = ./anno_sta/annos_objNum.png
get_relative = True
image_keyname = images
anno_keyname = annotations
Args_show = True

json read...

make dir: ./anno_sta
png save to ./anno_sta/annos_shape.png
png save to ./anno_sta/annos_shape_Relative.png
png save to ./anno_sta/annos_shapeRate.png
png save to ./anno_sta/annos_pos.png
png save to ./anno_sta/annos_pos_Relative.png
png save to ./anno_sta/annos_posEnd.png
png save to ./anno_sta/annos_posEnd_Relative.png
png save to ./anno_sta/annos_cat.png
png save to ./anno_sta/annos_objNum.png
csv save to ./anno_sta/annos.csv
```

部分表格内容：

![image.png](./assets/1650025881244-image.png)

所有目标检测框shape的二维分布：

![image.png](./assets/1650025909461-image.png)

所有目标检测框shape在图像中相对比例的二维分布：

![image.png](./assets/1650026052596-image.png)

所有目标检测框shape比例（宽/高）的一维分布：

![image.png](./assets/1650026072233-image.png)

所有目标检测框左上角坐标的二维分布：

![image.png](./assets/1650026247150-image.png)

所有目标检测框左上角坐标的相对比例值的二维分布：

![image.png](./assets/1650026289987-image.png)

所有目标检测框右下角坐标的二维分布：

![image.png](./assets/1650026457254-image.png)

所有目标检测框右下角坐标的相对比例值的二维分布：

![image.png](./assets/1650026487732-image.png)

各个类别的对象数量分布：

![image.png](./assets/1650026546304-image.png)

单个图像中含有标注对象的数量分布：

![image.png](./assets/1650026559309-image.png)

### 3.5 统计图像信息生成json

使用`json_Test2Json.py`，可以根据`test2017`中的文件信息与训练集json文件快速提取图像信息，生成测试集json文件。

#### 3.5.1 命令演示

执行如下命令，统计并生成`test2017`信息：

```
python ./coco_tools/json_Img2Json.py \
    --test_image_path=./test2017 \
    --json_train_path=./annotations/instances_val2017.json \
    --json_test_path=./test.json
```

#### 3.5.2 参数说明


| 参数名               | 含义                                      | 默认值        |
| ------------------- | ---------------------------------------- | ------------ |
| `--test_image_path` | 需要统计的图像目录路径                       |              |
| `--json_train_path` | 用于参考的训练集json文件路径                 |              |
| `--json_test_path`  | 生成的测试集json文件路径                    |              |
| `--image_keyname`   | （可选）json文件中，图像对应的key            | `'images'`    |
| `--cat_keyname`     | （可选）json文件中，类别对应的key            | `'categories'`|
| `--Args_show`       | （可选）是否打印输入参数信息                 | `True`        |

#### 3.5.3 结果展示

执行上述命令后，输出结果如下：

```
------------------------------------------------Args------------------------------------------------
test_image_path = ./test2017
json_train_path = ./annotations/instances_val2017.json
json_test_path = ./test.json
Args_show = True

----------------------------------------------Get Test----------------------------------------------

json read...

test image read...
100%|█████████████████████████████████████| 40670/40670 [06:48<00:00, 99.62it/s]

 total test image: 40670
```

生成的json文件信息：

```
------------------------------------------------Args------------------------------------------------
json_path = ./test.json
show_num = 5
Args_show = True

------------------------------------------------Info------------------------------------------------
json read...
json keys: dict_keys(['images', 'categories'])

**********************images**********************
 Content Type: list
 Total Length: 40670
 First 5 record:

{'id': 0, 'width': 640, 'height': 427, 'file_name': '000000379269.jpg'}
{'id': 1, 'width': 640, 'height': 360, 'file_name': '000000086462.jpg'}
{'id': 2, 'width': 640, 'height': 427, 'file_name': '000000176710.jpg'}
{'id': 3, 'width': 640, 'height': 426, 'file_name': '000000071106.jpg'}
{'id': 4, 'width': 596, 'height': 640, 'file_name': '000000251918.jpg'}
...
...

********************categories********************
 Content Type: list
 Total Length: 80
 First 5 record:

{'supercategory': 'person', 'id': 1, 'name': 'person'}
{'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}
{'supercategory': 'vehicle', 'id': 3, 'name': 'car'}
{'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}
{'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}
...
...
```

### 3.6 json文件拆分

使用`json_Split.py`，可以将`instances_val2017.json`文件拆分为2个子集。

#### 3.6.1 命令演示

执行如下命令，拆分`instances_val2017.json`文件：

```
python ./coco_tools/json_Split.py \
    --json_all_path=./annotations/instances_val2017.json \
    --json_train_path=./instances_val2017_train.json \
    --json_val_path=./instances_val2017_val.json
```

#### 3.6.2 参数说明


| 参数名                | 含义                                                                                   | 默认值        |
| -------------------- | ------------------------------------------------------------------------------------- | ------------ |
| `--json_all_path`    | 需要拆分的json文件路径                                                                   |              |
| `--json_train_path`  | 生成的train部分json文件                                                                 |              |
| `--json_val_path`    | 生成的val部分json文件                                                                   |              |
| `--val_split_rate`   | （可选）拆分过程中，val集文件的比例                                                        | `0.1`        |
| `--val_split_num`    | （可选）拆分过程中，val集文件的数量，<br />如果设置了该参数，则`--val_split_rate`参数失效       | `None`       |
| `--keep_val_inTrain` | （可选）拆分过程中，是否在train中仍然保留val部分                                            | `False`      |
| `--image_keyname`    | （可选）json文件中，图像对应的key                                                         | `'images'`    |
| `--cat_keyname`      | （可选）json文件中，类别对应的key                                                         | `'categories'`|
| `--Args_show`        | （可选）是否打印输入参数信息                                                              | `'True'`      |

#### 3.6.3 结果展示

执行上述命令后，输出结果如下：

```
------------------------------------------------Args------------------------------------------------
json_all_path = ./annotations/instances_val2017.json
json_train_path = ./instances_val2017_train.json
json_val_path = ./instances_val2017_val.json
val_split_rate = 0.1
val_split_num = None
keep_val_inTrain = False
image_keyname = images
anno_keyname = annotations
Args_show = True

-----------------------------------------------Split------------------------------------------------

json read...

image total 5000, train 4500, val 500
anno total 36781, train 33119, val 3662
```

### 3.7 json文件合并

使用`json_Merge.py`，可以合并2个json文件。

#### 3.7.1 命令演示

执行如下命令，合并`instances_train2017.json`与`instances_val2017.json`：

```
python ./coco_tools/json_Merge.py \
    --json1_path=./annotations/instances_train2017.json \
    --json2_path=./annotations/instances_val2017.json \
    --save_path=./instances_trainval2017.json
```

#### 3.7.2 参数说明


| 参数名          | 含义                             | 默认值                       |
| -------------- | ------------------------------- | --------------------------- |
| `--json1_path` | 需要合并的json文件1路径            |                             |
| `--json2_path` | 需要合并的json文件2路径            |                             |
| `--save_path`  | 生成的json文件                    |                             |
| `--merge_keys` | （可选）合并过程中需要合并的key      | `['images', 'annotations']` |
| `--Args_show`  | （可选）是否打印输入参数信息         | `True`                      |

#### 3.7.3 结果展示

执行上述命令后，输出结果如下：

```
------------------------------------------------Args------------------------------------------------
json1_path = ./annotations/instances_train2017.json
json2_path = ./annotations/instances_val2017.json
save_path = ./instances_trainval2017.json
merge_keys = ['images', 'annotations']
Args_show = True

-----------------------------------------------Merge------------------------------------------------

json read...

json merge...
info
licenses
images merge!
annotations merge!
categories

json save...

finish!
```
