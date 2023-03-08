from fileinput import close
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
    
def square(bbox, size):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1 + 1, y2 - y1 + 1
    if w < h:
        pad = (h - w) // 2
        x1 = max(0, x1 - pad)
        x2 = min(size[1], x2 + pad)
    else:
        pad = (w - h) // 2
        y1 = max(0, y1 - pad)
        y2 = min(size[0], y2 + pad)
    return x1, y1, x2, y2


def pad(bbox, img_size, pad_scale=0.5):
    """pad bbox with scale
    Args:
        bbox (list):[x1, y1, x2, y2]
        img_size (tuple): (height, width)
        pad_scale (float): scale for padding
    Return:
        bbox (list)
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1 + 1, y2 - y1 + 1
    dw = int(w * pad_scale)
    dh = int(h * pad_scale)
    x1 = max(0, x1 - dw)
    x2 = min(img_size[1], x2 + dw)
    y1 = max(0, y1 - dh)
    y2 = min(img_size[0], y2 + dh)
    return x1, y1, x2, y2


def adjust_bbox(bbox, img_shape, pad_scale=0.5):
    bbox = square(bbox, img_shape)
    bbox = pad(bbox, img_shape, pad_scale)
    return bbox

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
    dataset_split = 'train'
    image_path = './dataset/kolektor2/images/train/'
    anno_path = './dataset/kolektor2/anno/train/'
    output_path = './dataset/kolektor2/ROI/images/train/'
    output_anno_path = './dataset/kolektor2/ROI/anno/train/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_anno_path):
        os.makedirs(output_anno_path)
    images = []
    annotations = []
    need_mask = True
    image_id = 0
    anno_id = 0
    classid_list = [1, 2, 3, 4]
    file_list = os.path.join('./dataset/kolektor2/ROI/', dataset_split + '.txt')
    f = open(file_list, "w")

    for img_name in os.listdir(image_path):
        image_id += 1
        file_name = os.path.join(image_path, img_name)
        img = cv2.imread(file_name)

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
                
                bbox = adjust_bbox([int(stat[0]), int(stat[1]), int(stat[0]+stat[2]), int(stat[1]+stat[3])], img.shape[:2])
                img_crop = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
                mask_crop = mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                #import pdb;pdb.set_trace()
                cv2.imwrite(f'./{output_path}{img_name[:-4]}_{cls_id}_{i}.png',img_crop)
                cv2.imwrite(f'./{output_anno_path}{img_name[:-4]}_{cls_id}_{i}_GT.png',mask_crop)
                left = f'./{output_path}{img_name[:-4]}_{cls_id}_{i}.png'.replace('./dataset/kolektor2/ROI/','')
                right = f'./{output_anno_path}{img_name[:-4]}_{cls_id}_{i}_GT.png'.replace('./dataset/kolektor2/ROI/','')
                line = left + " " + right + '\n'
                f.write(line)
    f.close()