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

import math
import os
from collections import defaultdict

import numpy as np

import paddle
import ppindustry
from ppindustry.cvlib.workspace import create
from ppindustry.ops import *
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

    def update_res(self, results, op_outputs, input_name):
        # step1: remove the result when keys not used in later input
        for res, out in zip(results, op_outputs):
            if self.has_output_op:
                del_name = []
                for k in out.keys():
                    if k not in self.input_dep:
                        del_name.append(k)
                # remove the result when keys not used in later input
                for name in del_name:
                    del out[name]
            res.update(out)

        # step2: if the input name is no longer used, then result will be deleted  
        if self.has_output_op:
            for name in input_name:
                self.input_dep[name] -= 1
                if self.input_dep[name] == 0:
                    for res in results:
                        del res[name]

    def run(self, input, frame_id=-1):
        # execute each operator according to toposort order
        for op_name, op in self.op_name2op.items():
            result = op(input)
            input = result
        print(result)
        return result
