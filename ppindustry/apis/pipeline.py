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

import os
import sys
import numpy as np
import math
import glob
import paddle
import cv2
from collections import defaultdict
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from ppindustry.cvlib.framework import Builder
from ppindustry.utils.logger import setup_logger
from ppindustry.cvlib.configs import ConfigParser

logger = setup_logger('pipeline')

__all__ = ['Pipeline']


class Pipeline(object):
    def __init__(self, cfg):
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()
        self.exe = Builder(self.model_cfg, self.env_cfg)
        self.output_dir = self.env_cfg.get('output_dir', 'output')

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

    def run(self, input):
        input, input_type = self._parse_input(input)
        if input_type == "image" :
            results = self.predict_images(input)
        elif input_type == "json":
            results = self.predict_json(input)
        else:
            raise ValueError("Unexpected input type: {}".format(input_type))
        return results

    def decode_image(self, input):
        if isinstance(input, str):
            with open(input, 'rb') as f:
                im_read = f.read()
            data = np.frombuffer(im_read, dtype='uint8')
            im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = input
        return im

    def predict_images(self, input):
        results = self.exe.run(input)
        return results

    def predict_json(self, input):
        results = {}
        logger.info('save result to {}'.format(input))

        return results
