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

# 测试
if __name__ == "__main__":
    model = VGGFeatureExtractor()
    input_tensor = torch.randn(1, 3, 256, 256)  # 假设输入为一张 3 通道的 256x256 图像
    output = model(input_tensor)
    print("VGG 特征提取输出维度:", output.shape)