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
from ppindustry.utils.visualizer import show_result
from ppdet.utils.colormap import colormap


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
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
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

    def run_ranks(self, input):
        image_list, input_type = self._parse_input(input)
        
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if nranks > 1:
            img_lists = self.partition_list(image_list, nranks)
        else:
            img_lists = [image_list]
        results = []

        for i, im_data in enumerate(img_lists[local_rank]):
            output = self.predict_images(im_data)
            results.append(output)
            logger.info(
                'processed the images automatically')
        all_results = []
        if local_rank == 0:
            paddle.distributed.all_gather_object(all_results, results)
        

        return results

    def run(self, input):
        input, input_type = self._parse_input(input)
        if input_type == "image" :
            results = self.predict_images(input)
        else:
            raise ValueError("Unexpected input type: {}".format(input_type))
        return results
        


    def predict_images(self, input):
        results = self.modules.run(input)
        #results = self.update(results, input)
        if self.vis:
            show_result(results, self.output_dir)

        if self.save: 
            with open(os.path.join(self.output_dir, 'output.json'), "w") as f: 
                json.dump(results, f, indent=2) 

        return results

