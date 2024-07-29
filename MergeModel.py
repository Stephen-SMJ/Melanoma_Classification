import torch
#方案2
from torch import nn

# -*- coding:utf8 -*-
import torch
import torch.nn.functional as F
from torch import nn

class MergeModel(nn.Module):
    def __init__(self, model1, model2): #model：增强alexnet model2:下采样alexnet
        super(MergeModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.Layers = nn.Sequential(
            nn.Linear(4,16),
            DropBlock(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16,2)
        )
    def forward(self, images) -> torch.Tensor:
        out1 = self.model1(images)
        out2 = self.model2(images)
        res = torch.cat((out1, out2), axis=-1)
        outputs = self.Layers(res)
        return outputs


class DropBlock(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if self.drop_prob == 0:
            return x
        # 设置gamma,比gamma小的设置为1,大于gamma的为0,对应第五步
        # 这样计算可以得到丢弃的比率的随机点个数
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        mask = mask.to(x.device)

        # compute block mask
        block_mask = self._compute_block_mask(mask)
        # apply block mask,为算法图的第六步
        out = x * block_mask[:, None, :, :]
        # Normalize the features,对应第七步
        out = out * block_mask.numel() / block_mask.sum()
        return out

    def _compute_block_mask(self, mask):
        # 取最大值,这样就能够取出一个block的块大小的1作为drop,当然需要翻转大小,使得1为0,0为1
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask


# #方案1
# class MergeModel():
#     def __init__(self, model1, model2): #model：增强alexnet model2:下采样net
#         self.model1 = model1
#         self.model2 = model2
#     def output(self,images):
#         out1 = self.model1(images)
#         out2 = self.model2(images)
#         _, predicted1 = torch.max(out1.data, 1)
#         _, predicted2 = torch.max(out2.data, 1)
#         outputs = []
#         for i in range(len(predicted1)):
#             if predicted1[i] == 0 and predicted2[i] == 0:
#                 output = 0
#             elif predicted1[i] == 0 and predicted2[i] == 1:
#                 output = 1
#             elif predicted2[i] == 1 and predicted2[i] == 1:
#                 output = 1
#             else:
#                 output = 0
#             outputs.append(output)
#         outputs = torch.tensor(outputs)
#         return outputs


