import os
import json
import argparse

import cv2
import numpy as np
import pycocotools.mask as mask_util
# 'x, y, w, h = cv2.boundingRect(mask)'
'''
def get_bounding_boxes(self) :
    """
    Returns:
        Boxes: tight bounding boxes around bitmasks.
        If a mask is empty, it's bounding box will be all zero.
    """
    boxes = torch.zeros(self.tensor.shape[0], 4, dtype=torch.float32)
    x_any = torch.any(self.tensor, dim=1)
    y_any = torch.any(self.tensor, dim=2)
    for idx in range(self.tensor.shape[0]):
        x = torch.where(x_any[idx, :])[0]
        y = torch.where(y_any[idx, :])[0]
        if len(x) > 0 and len(y) > 0:
            boxes[idx, :] = torch.as_tensor(
                [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
            )
    return Boxes(boxes)
'''
def conver_mask(image_path, mask_path):
    pass
    
def label_to_onehot(label, classid_list):  
    """  
    Converts a segmentation label (H, W, C) to (H, W, K) where the last dim is a one  
    hot encoding vector, C is usually 1 or 3, and K is the number of class.  
    """  
    semantic_map = []  
    for cls_id in classid_list:  
        equality = np.equal(label, cls_id)  
        class_map = np.all(equality, axis=-1)  
        ret, labels, stats, centroid = cv2.connectedComponentsWithStats(class_map)
        semantic_map.append(class_map)  
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)  
    return semantic_map 

def get_args():
    parser = argparse.ArgumentParser(
        description='Mask Format convert to Json for detection ')

    # Parameters
    parser.add_argument(
        '--image_path',
        type=str,
        help='json path to statistic images information')
    parser.add_argument(
        '--mask_path',
        type=str,
        help='ground truth path'
    )
    parser.add_argument(
        '--classid',
        type=str,
        default=None,
        help='classname of mask if classid is not None, otherwise mask format will be saved'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help='save path to save coco format json'
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    image_path = './dataset/kolektor2/images/train/'
    anno_path = './dataset/kolektor2/anno/train/'
    output_path = './dataset/kolektor2/train.json'
    images = []
    annotations = []
    need_mask = True
    image_id = 0
    anno_id = 0
    classid_list = [1, 2, 3, 4]
    for img_name in os.listdir(image_path):
        image_id += 1
        file_name = os.path.join(image_path, img_name)
        img = cv2.imread(file_name)
        image_info = {'file_name': file_name,
                    'id': image_id,
                    'width': img.shape[1],
                    'height': img.shape[0],
        }
        images.append(image_info)

        anno_name = os.path.join(anno_path, img_name[:-4]+'_GT.png')
        mask = cv2.imread(anno_name, -1)
        if mask.sum() == 0: 
            continue
    
        for cls_id in classid_list:  
            class_map = np.equal(mask, cls_id).astype(np.uint8)
            if class_map.sum()==0:
                continue
            
            ret, labels, stats, centroid = cv2.connectedComponentsWithStats(class_map)
            #import pdb; pdb.set_trace()
            for i, stat in enumerate(stats):
                if i == 0:
                    continue
                anno = {}
                polygon = []

                if need_mask:
                    contours, _ = cv2.findContours((labels==i).astype(np.uint8), cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
                    polygon.append(contours[0].flatten().astype(np.uint8).tolist())
    
                anno_id += 1
                anno = {'id': anno_id,
                'image_id': image_id,
                'category_id': cls_id,
                "segmentation": polygon,
                "area": int(stat[2])*int(stat[3]), 
                "bbox": [int(stat[0]), int(stat[1]), int(stat[2]), int(stat[3])],
                "iscrowd": 0,     
                }
                annotations.append(anno)
                

    categories = [{'supercategory': 'defect',
                    'id': 1, 
                    'name':'ys'}]

    json_data = {'images': images, 'annotations':annotations, 'categories':categories}
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)