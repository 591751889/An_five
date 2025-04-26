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


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )

        # 分类层（输出logits）
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        # 输入形状: [B, C, L] = [8, 3, 256]
        x = x.mean(dim=1)  # 通道维度聚合 [8, 256]
        x = self.feature_net(x)  # [8, 128]
        return self.classifier(x)  # 输出logits [8, 3]

class AnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_layers = AttentionFusion()

        self.classificationModel = ClassificationModel()

    def forward(self, x):
        # 输入形状: [8, 3, 256]
        print(x.size())
        x = self.attention_layers(x)
        return self.classificationModel(x)
# 测试
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 3, 256, 256)  # 图像数据
        self.labels = torch.randint(0, 2, (num_samples,)).float()  # 二分类标签

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_dataset = ImageDataset(800)
    val_dataset = ImageDataset(200)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # 模型初始化
    model = AnModel().to(device)
    criterion = nn.BCEWithLogitsLoss()  # 二分类损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(10):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze().float()
# 从[B,1] -> [B]
            print(outputs.shape,labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs).squeeze()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1}/10 | Loss: {running_loss / len(train_loader):.4f} | "
              f"Val Acc: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    # 验证模型维度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = AnModel().to(device)


    test_input = torch.randn(8, 3, 256, 256).to(device)
    output = test_model(test_input)
    print("输出维度:", output.shape)  # 应为[8, 1]

    # 开始训练
    train_model()