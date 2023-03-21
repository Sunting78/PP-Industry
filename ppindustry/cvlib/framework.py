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

import math
import os
from collections import defaultdict

import numpy as np

import paddle
import ppindustry
from ppindustry.cvlib.workspace import create
from ppindustry.ops import *
from ppindustry.utils.data_dict import post_process_image_info
from ppindustry.utils.helper import gen_input_name, get_output_keys


class Builder(object):
    """
    The executor which implements model series pipeline

    Args:
        env_cfg: The enrionment configuration
        model_cfg: The models configuration
    """

    def __init__(self, model_cfg, env_cfg=None):
        self.model_cfg = model_cfg
        self.op_name2op = {}
        self.has_output_op = False
        for op in model_cfg:
            op_arch = list(op.keys())[0]
            op_cfg = list(op.values())[0]
            op = create(op_arch, op_cfg, env_cfg)
            self.op_name2op[op_arch] = op


    def update(self, results, input):
        """"update model results"""
        image_to_info = {}
        for data_path in input:
            image_to_info[data_path] = {'pred': []}
        for pred in results:
            image_path = pred['image_path']
            pred.pop("image_path")
            pred.pop("image_id") if "image_id" in pred.keys() else None
            if  image_path not in image_to_info.keys():
                image_to_info[image_path]= {'pred':[pred]}
            
            else:
                image_to_info[image_path]['pred'].append(pred)
            
            #if 'isNG' not in image_to_info[image_path].keys() or image_to_info[image_path]['isNG'] == 0:
            #    image_to_info[image_path]['isNG'] = pred['isNG']

            #if not image_to_info[image_path].get('isNG', 0):
            #    image_to_info[image_path]['isNG'] = pred['isNG']

        return image_to_info



    def run(self, input, frame_id=-1):
        image_list = input
        # execute each operator according to toposort order
        for op_name, op in self.op_name2op.items():
            if op_name == 'PostProcess':
                input = self.update(result, image_list)
            result = op(input)
            input = result
        
        post_process_image_info(result)

            
        print(result)
        return result
