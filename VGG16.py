import torch
from torch import nn

import torchvision.models as models


# class VGGnet(nn.Module):
#     def __init__(self, feature_extract=True, num_classes=3):
#         super(VGGnet, self).__init__()
#         model = models.vgg16_bn(pretrained=True)
#         self.features = model.features
#         set_parameter_requires_grad(self.features, feature_extract)  # 固定特征提取层参数
#         # 自适应输出宽高尺寸为7×7
#         self.avgpool = model.avgpool
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         #        print(x.shape)
#         x = x.view(x.size(0), 512 * 7 * 7)
#         out = self.classifier(x)
#         return out
#
# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False


class VGG(nn.Module):

    def __init__(
        self,
        # features: nn.Module,
        num_classes: int = 2,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        model = models.vgg16_bn(pretrained=True)
        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)