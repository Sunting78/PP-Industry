# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import numpy as np

import cv2
import paddle
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.core.predict import mkdir, partition_list
from paddleseg.cvlibs import Config
from paddleseg.transforms import Compose
from paddleseg.utils import progbar, visualize
from ppindustry.utils.logger import setup_logger

logger = setup_logger('SegPredictor')

def preprocess(im_data, transforms):
    
    if not isinstance(im_data, dict):
        data = {}
        data['img'] = im_data
    else:
        data = im_data
    data = transforms(data)
    data['img'] = data['img'][np.newaxis, ...]
    data['img'] = paddle.to_tensor(data['img'])
    return data

class SegPredictor(object):
    def __init__(self, seg_config, seg_model):
        self.seg_config = seg_config
        self.model = seg_config.model
        self.model_path = seg_model
        self.visualize = False
        self.transforms = Compose(seg_config.val_transforms)
        utils.utils.load_entire_model(self.model, self.model_path)
        self.model.eval()
    
    def postprocess(self, pred, img_data):
        class_list = np.unique(pred)
        result = []
        # get polygon
        
        for cls_id in class_list:
            
            if cls_id == 0:
                continue # skip background
            class_map = np.equal(pred, cls_id).astype(np.uint8)
            contours, _ = cv2.findContours(class_map, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
            polygon = []
            for j in range(len(contours)):
                polygon.append(contours[j].flatten().tolist())
            
            result.append({
                'img_path': img_data,
                'category_id': cls_id,
                #'mask': class_map,
                'polygon': polygon,
                'area': np.sum(class_map > 0),
            })


        return result

    def roi_postprocess(self, pred, img_data):
        
        class_list = np.unique(pred)
        result = []
        for cls_id in class_list:
            if cls_id == 0:
                continue # skip background
            class_map = np.equal(pred, cls_id).astype(np.uint8)

            crop_bbox = img_data['crop_bbox']
            bbox = img_data['bbox']
            # get mask
            offset_left = bbox[0] - crop_bbox[0]
            offset_top = bbox[1] - crop_bbox[1]
            offset_right = offset_left + bbox[2] - bbox[0]
            offset_bottom = offset_top + bbox[3] - bbox[1]
            class_map = class_map[int(offset_top):int(offset_bottom),
                             int(offset_left):int(offset_right)].astype(np.uint8)
            contours, _ = cv2.findContours(class_map, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
            polygon = []
            for j in range(len(contours)):
                contours[j][..., 0] += int(bbox[0])
                contours[j][..., 1] += int(bbox[1])
                polygon.append(contours[j].flatten().tolist())
            img_data.pop('img', None)
            #img_data.pop('crop_box', None)
            img_data['polygon'] = polygon
            img_data['area'] =  np.sum(class_map > 0)

            result.append(img_data)

        return result

    def predict(self,
                image_list,
                image_dir=None,
                save_dir='output',
                aug_pred=False,
                scales=1.0,
                flip_horizontal=True,
                flip_vertical=False,
                is_slide=False,
                stride=None,
                crop_size=None,
                custom_color=None):

        results = []
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if nranks > 1:
            img_lists = partition_list(image_list, nranks)
        else:
            img_lists = [image_list]

        added_saved_dir = os.path.join(save_dir, 'added_prediction')
        pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')

        logger.info("Start to predict...")
        progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
        color_map = visualize.get_color_map_list(256, custom_color=custom_color)
        with paddle.no_grad():
            for i, im_data in enumerate(img_lists[local_rank]):
                data = preprocess(im_data, self.transforms)
                if aug_pred:
                    pred, _ = infer.aug_inference(
                        self.model,
                        data['img'],
                        trans_info=data['trans_info'],
                        scales=scales,
                        flip_horizontal=flip_horizontal,
                        flip_vertical=flip_vertical,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                else:
                    pred, _ = infer.inference(
                        self.model,
                        data['img'],
                        trans_info=data['trans_info'],
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                pred = paddle.squeeze(pred)
                pred = pred.numpy().astype('uint8')

                if isinstance(im_data, dict):
                    result = self.roi_postprocess(pred, im_data)
                else:
                    result = self.postprocess(pred, im_data)
                results.extend(result)


                # save added image
                if self.visualize:
                    # get the saved name
                    if image_dir is not None:
                        im_file = im_path.replace(image_dir, '')
                    else:
                        im_file = os.path.basename(im_path)
                    if im_file[0] == '/' or im_file[0] == '\\':
                        im_file = im_file[1:]

                    added_image = utils.visualize.visualize(
                        im_path, pred, color_map, weight=0.6)
                    added_image_path = os.path.join(added_saved_dir, im_file)
                    mkdir(added_image_path)
                    cv2.imwrite(added_image_path, added_image)

                    # save pseudo color prediction
                    pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
                    pred_saved_path = os.path.join(
                        pred_saved_dir, os.path.splitext(im_file)[0] + ".png")
                    mkdir(pred_saved_path)
                    pred_mask.save(pred_saved_path)

                progbar_pred.update(i + 1)
        return results
