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

from functools import reduce
import os
import importlib
import numpy as np
import math
import paddle

from ppindustry.cvlib.workspace import register
from ppindustry.cvlib.workspace import create
from ppindustry.utils.logger import setup_logger
logger = setup_logger('PostProcess')

@register
class PostProcess(object):
    def __init__(self, model_cfg, env_cfg):
        super(PostProcess, self).__init__()
        #self.model_cfg = model_cfg
        self.env_cfg = env_cfg
        self.op_name2op = {}
        self.rule = []
        self.init(model_cfg)


    def init(self, cfg):
        self.rules = []
        import pdb;pdb.set_trace()
        if isinstance(cfg, list):
            for sub_op in cfg:
                sub_op_arch = list(sub_op.keys())[0]
                sub_op_cfg = list(sub_op.values())[0]
                sub_op = create(sub_op_arch, sub_op_cfg, self.env_cfg)
                self.op_name2op[sub_op_arch] = sub_op
                self.rules.append(sub_op)
        logger.info("PostProcess has {} rules".format(
            len(self.rules)))

    def __len__(self):
        return len(self.rules)

    def __call__(self, input):
        for rule in self.rules:
            result = rule(input)
            input = result
        return result
       

@register
class JudgeDetByScores(object):
    def __init__(self, cfg, env_cfg=None):
        super(JudgeDetByScores, self).__init__()
        self.score_threshold = cfg['score_threshold']
        #mod = importlib.import_module(__name__)

    def __call__(self, input):
        
        import pdb;pdb.set_trace()
        
        return input

@register
class JudgeByLengthWidth(object):
    def __init__(self, cfg, env_cfg=None):
        super(JudgeByLengthWidth, self).__init__()
        self.filterlen_thresh = cfg['filterlen_thresh']
        #mod = importlib.import_module(__name__)

    def __call__(self, input):
        import pdb;pdb.set_trace()
        return input

@register
class JudgeByArea(object):
    def __init__(self, cfg, env_cfg=None):
        super(JudgeByArea, self).__init__()
        self.model_cfg = cfg
        #mod = importlib.import_module(__name__)

    def __call__(self, input):
        import pdb;pdb.set_trace()
        return input