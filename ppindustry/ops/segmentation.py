# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import importlib
import math
import os
from functools import reduce

import numpy as np

import cv2
import paddle
from paddleseg.cvlibs import Config
from paddleseg.utils import get_image_list, get_sys_env, logger
from ppindustry.cvlib.workspace import register
from ppindustry.seg.engine import SegPredictor


@register
class BaseSegmentation(object):
    def __init__(self, model_cfg, env_cfg):
        super(BaseSegmentation, self).__init__()
        seg_config = model_cfg['config_path']
        seg_config = Config(seg_config)
        seg_model = model_cfg['model_path']
        self.predictor = SegPredictor(seg_config, seg_model)   


    def __call__(self, inputs):
        results = self.predictor.predict(image_list = inputs)
        
        return results

@register
class CropSegmentation(object):
    def __init__(self, model_cfg, env_cfg):
        super(CropSegmentation, self).__init__()
        seg_config = model_cfg['config_path']
        seg_config = Config(seg_config)
        seg_model = model_cfg['model_path']
        self.crop_score_thresh = model_cfg['crop_score_thresh']
        self.pad_scale = model_cfg['pad_scale']

        self.predictor = SegPredictor(seg_config, seg_model) 

    def square(self, bbox, size):
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


    def pad(self, bbox, img_size, pad_scale=0.0):
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

    def adjust_bbox(self, bbox, img_shape, pad_scale=0.0):
        bbox = self.square(bbox, img_shape)
        bbox = self.pad(bbox, img_shape, pad_scale)
        return bbox

    def __call__(self, input):
        for data in input:
            image_path = data['image_path']
            bbox =  data['bbox']
            img = cv2.imread(image_path)
            crop_bbox = self.adjust_bbox(
                                        [int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])], 
                                        img_shape=img.shape[:2], 
                                        pad_scale=self.pad_scale)

            img_crop = img[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
            data['img'] = img_crop
            data['img_shape'] = img.shape[:2]
            data['crop_bbox'] = crop_bbox
        results = self.predictor.predict(input)


        return results
