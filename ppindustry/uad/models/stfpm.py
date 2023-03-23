import paddle
import paddle.nn as nn
from paddle.vision.models.resnet import resnet18, resnet34, resnet50, resnet101

from ppindustry.cvlib.workspace import register


models = {"resnet18": resnet18, "resnet34": resnet34,
          "resnet50": resnet50, "resnet101": resnet101,}

@register
class ResNet_MS3(nn.Layer):
    def __init__(self, arch='resnet18', pretrained=True):
        super(ResNet_MS3, self).__init__()
        assert arch in models.keys(), 'arch {} not supported'.format(arch)
        net = models[arch](pretrained=pretrained)
        # ignore the last block and fc
        self.model = paddle.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in self.model._sub_layers.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res


class ResNet_MS3_EXPORT(nn.Layer):

    def __init__(self, student, teacher):
        super(ResNet_MS3_EXPORT, self).__init__()
        self.student = student
        self.teacher = teacher

    def forward(self, x):
        result = []
        result.append(self.student(x))
        result.append(self.teacher(x))
        return result