import torch
import torch.nn as nn
import torchvision.models as models

class AlexNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(AlexNetFeatureExtractor, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        # 移除最后一层全连接层，保留特征提取部分
        self.feature_extractor = self.alexnet.features
        # 添加全局平均池化层，确保输出特征维度一致
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 添加一个全连接层来调整输出维度
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # 移除最后一层全连接层，保留特征提取部分
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        # 添加全局平均池化层，确保输出特征维度一致
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 256)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        # 将特征展平
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


import torch
import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        # 移除最后一层全连接层，保留特征提取部分
        self.feature_extractor = self.vgg.features
        # 添加全局平均池化层，确保输出特征维度一致
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 添加一个全连接层来调整输出维度
        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
import torch
from torch import nn
from torch.nn import init

# "Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks"
class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model

        return out


class AttentionFusion(nn.Module):
    def __init__(self):
        super(AttentionFusion, self).__init__()
        # 创建三个特征提取器实例
        self.resnet_extractor = ResNetFeatureExtractor()
        self.vgg_extractor = VGGFeatureExtractor()
        self.alexnet_extractor = AlexNetFeatureExtractor()

        # 注意力机制相关层
        self.attention_layers = ExternalAttention(d_model=256, S=8)


    def forward(self, x):
        # 提取每个特征提取器的特征
        resnet_features = self.resnet_extractor(x)
        vgg_features = self.vgg_extractor(x)
        alexnet_features = self.alexnet_extractor(x)

        # 将特征堆叠起来
        stacked_features = torch.stack([resnet_features, vgg_features, alexnet_features], dim=1)
        print(stacked_features.size())



        attention = self.attention_layers(stacked_features)
        # 计算注意力权重


        return attention


# 测试
if __name__ == "__main__":
    model = AttentionFusion()
    input_tensor = torch.randn(8, 3, 256, 256)  # 假设输入为一张3通道的256x256图像
    output = model(input_tensor)
    print("融合后的特征维度:", output.shape)