# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from random import sample
from collections import OrderedDict
from scipy.ndimage import gaussian_filter
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.vision.models.resnet import ResNet, BasicBlock, BottleneckBlock
from paddle.vision.models.resnet import model_urls,get_weights_path_from_url

from ppindustry.cvlib.workspace import register

@register
class ResNet_PaDiM(nn.Layer):
    def __init__(self, depth=18, pretrained=True):
        super(ResNet_PaDiM, self).__init__()
        arch = 'resnet{}'.format(depth)
        Block = BottleneckBlock
        if depth < 50:
            Block = BasicBlock
        self.model = ResNet(Block, depth)
        self.distribution = None
        self.init_weight(arch, pretrained)

    def init_weight(self, arch, pretrained):
        if pretrained:
            assert arch in model_urls, "{} model do not have a pretrained model now," \
                                       " you should set pretrained=False".format(arch)
            weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                    model_urls[arch][1])

            self.model.set_dict(paddle.load(weight_path))
            print("Successfully load backbone pretrained weights.")

    def forward(self, x):
        res = []
        with paddle.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            res.append(x)
            x = self.model.layer2(x)
            res.append(x)
            x = self.model.layer3(x)
            res.append(x)
        return res


class PaDiM_Export(nn.Layer):
    def __init__(self, depth=18, pretrained=True, t_d=448, d=100):
        super(PaDiM_Export, self).__init__()
        arch = 'resnet{}'.format(depth)
        Block = BottleneckBlock
        if depth < 50:
            Block = BasicBlock
        self.model = ResNet(Block, depth)
        self.distribution = None
        self.init_weight(arch, pretrained)
        self.idx = paddle.to_tensor(sample(range(0, t_d), d))

    def init_weight(self, arch, pretrained):
        if pretrained:
            assert arch in model_urls, "{} model do not have a pretrained model now," \
                                       " you should set pretrained=False".format(arch)
            weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                    model_urls[arch][1])

            self.model.set_dict(paddle.load(weight_path))

    def forward(self, x):
        ori_shape = x.shape
        res = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        res.append(x)
        x = self.model.layer2(x)
        res.append(x)
        x = self.model.layer3(x)
        res.append(x)
        self.generate_scores_map(res, ori_shape)

    def generate_scores_map(self, outputs, ori_shape):
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.detach())
        for k, v in test_outputs.items():
            test_outputs[k] = paddle.concat(v, 0)

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            layer_embedding = test_outputs[layer_name]
            layer_embedding = F.interpolate(layer_embedding, size=embedding_vectors.shape[-2:], mode="nearest")
            embedding_vectors = paddle.concat((embedding_vectors, layer_embedding), 1)

        # randomly select d dimension
        embedding_vectors = paddle.index_select(embedding_vectors, self.idx, 1)

        # calculate distance matrix
        B, C, H, W = embedding_vectors.shape
        embedding = embedding_vectors.reshape((B, C, H * W))
        # calculate mahalanobis distances
        mean, covariance = paddle.to_tensor(self.distribution[0]), paddle.to_tensor(self.distribution[1])
        inv_covariance = paddle.linalg.inv(covariance.transpose((2, 0, 1)))

        delta = (embedding - mean).transpose((2, 0, 1))

        distances = (paddle.matmul(delta, inv_covariance) * delta).sum(2).transpose((1, 0))
        distances = distances.reshape((B, H, W))
        distances = paddle.sqrt(distances)
        score_map = F.interpolate(distances.unsqueeze(1), size=x.shape[2:], mode='bilinear',
                                align_corners=False).squeeze(1).numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        return scores