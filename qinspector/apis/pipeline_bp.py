# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
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

import glob
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import paddle

from ppindustry.cvlib.configs import ConfigParser
from ppindustry.cvlib.framework import Builder
from ppindustry.utils.logger import setup_logger
from ppindustry.utils.visualizer import draw_bbox
import pycocotools.mask as mask_util
from ppdet.utils.colormap import colormap

def polygons_to_bitmask(polygons, height, width):
    if len(polygons) == 0 :
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(int)

    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)

    return mask_util.decode(rle).astype(int)

logger = setup_logger('pipeline')

class Pipeline(object):
    def __init__(self, cfg):
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        self.modules = Builder(self.model_cfg, self.env_cfg)
        self.output_dir = self.env_cfg.get('output_dir', 'output')
        self.vis = self.env_cfg.get('visualize', False)
        self.save = self.env_cfg.get('save', False)

    def _parse_input(self, input):
        im_exts = ['jpg', 'jpeg', 'png', 'bmp']
        im_exts += [ext.upper() for ext in im_exts]
        json_exts = ['json']

        if isinstance(input, (list, tuple)) and isinstance(input[0], str):
            input_type = "image"
            images = [
                image for image in input
                if any([image.endswith(ext) for ext in im_exts])
            ]
            assert len(images) > 0, "no image found"
            logger.info("Found {} inference images in total.".format(len(images)))
            return images, input_type

        if os.path.isdir(input):
            input_type = "image"
            logger.info(
                'Input path is directory, search the images automatically')
            images = set()
            infer_dir = os.path.abspath(input)
            for ext in im_exts:
                images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
            images = list(images)
            assert len(images) > 0, "no image found"
            logger.info("Found {} inference images in total.".format(len(images)))
            return images, input_type

        logger.info('Input path is {}'.format(input))
        input_ext = os.path.splitext(input)[-1][1:]
        if input_ext in im_exts:
            input_type = "image"
            return [input], input_type

        if input_ext in json_exts:
            input_type = "json"
            return [input], input_type
        
        raise ValueError("Unsupported input format: {}".fomat(input_ext))
        return

    def partition_list(aself, arr, m):
        """split the list 'arr' into m pieces"""
        image_list = []
        for idx in range(m):
            slen = len(arr) // m
            start = idx * slen
            if idx == m - 1:
                end = len(arr)
            else:
                end = (idx + 1) * slen
            image_list.append(arr[start:end])
        return image_list

    def run(self, input):
        image_list, input_type = self._parse_input(input)
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if nranks > 1:
            img_lists = self.partition_list(image_list, nranks)
        else:
            img_lists = [image_list]
        results = []
        for i, im_data in enumerate(img_lists[local_rank]):
            data_dict = {}
            output = self.predict_images(im_data)
            results.append(output)
            logger.info(
                'processed the images automatically')
        all_results = []
        if local_rank == 0:
            paddle.distributed.all_gather_object(all_results, results)
        return results

    def update(self, results):
        """"update model results"""
        outputs = []
        image_to_info = {}
        for pred in results:
            image_path = pred['image_path']
            pred.pop("image_path")
            import pdb;pdb.set_trace()
            if  image_path not in image_to_info.keys():
                image_to_info[image_path]= {'pred':[pred]}
            
            else:
                image_to_info[image_path]['pred'].append(pred)
            
            #if 'isNG' not in image_to_info[image_path].keys() or image_to_info[image_path]['isNG'] == 0:
            #    image_to_info[image_path]['isNG'] = pred['isNG']

            if not image_to_info[image_path].get('isNG', 0):
                image_to_info[image_path]['isNG'] = pred['isNG']

        return image_to_info
        

    def show_result(self, results):
        catid2color = {}
        for im_path, preds in results.items():
            im_file = os.path.basename(im_path)

            image = Image.open(im_path)
            num = 0
            
            color_list = colormap(rgb=True)[:40]
            for pred in preds['pred']:
                draw = ImageDraw.Draw(image)
                cate_id = pred['category_id']
                if cate_id not in catid2color:
                    idx = np.random.randint(len(color_list))
                    catid2color[cate_id] = color_list[idx]
                color = tuple(catid2color[cate_id])

                if 'bbox' in pred:
                    bbox  = pred['bbox']
                    xmin, ymin, w, h = bbox
                    xmax = xmin + w
                    ymax = ymin + h
                    draw.line(
                        [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                        (xmin, ymin)],
                        width=2,
                        fill=color)

                    isNG = pred['isNG']
                    if 'score' in pred:
                        score = pred['score']
                        text = "{} {:.2f} {}".format(str(cate_id), score, 'NG' if isNG else 'OK')
                    else:
                        text = "{} {}".format(str(cate_id), 'NG' if isNG else 'OK')
                    
                    tw, th = draw.textsize(text)
                    if ymin - th >=1:
                        draw.rectangle(
                            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
                        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
                    else:
                        draw.rectangle(
                            [(xmin + 1, ymax ), (xmin + tw + 1, ymax + th)], fill=color)
                        draw.text((xmin + 1, ymax + 1), text, fill=(255, 255, 255)) 

                if 'polygon' in pred:
                    polygons = pred['polygon']
                    if len(polygons) == 0 or len(polygons[0]) < 4:
                        continue
                    alpha = 0.7
                    w_ratio = .4
                    img_array = np.array(image).astype('float32')
                    color = np.asarray(color)
                    for c in range(3):
                        color[c] = color[c] * (1 - w_ratio) + w_ratio * 255
                    
                    mask = polygons_to_bitmask(polygons, img_array.shape[0], img_array.shape[1])

                    idx = np.nonzero(mask)
                    img_array[idx[0], idx[1], :] *= 1.0 - alpha
                    img_array[idx[0], idx[1], :] += alpha * color
                    image = Image.fromarray(img_array.astype('uint8'))

            pred_saved_path = os.path.join(
                self.output_dir, os.path.splitext(im_file)[0] + ".png")
            image.save(pred_saved_path)


    def predict_images(self, input):
        import pdb;pdb.set_trace()
        results = self.modules.run(input)
        import pdb;pdb.set_trace()
        results = self.update(results)
        if self.vis:
            self.show_result(results)
        if self.save: 
            with open(os.path.join(self.output_dir, 'output.json'), "w") as f: 
                json.dump(results, f, indent=2) 


        

        return results

