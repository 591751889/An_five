import torch
import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 全局平均池化：将长度维度256 -> 1，输出形状 [8, 3, 1]
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Softmax在通道维度（dim=1）计算概率
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 输入形状: [8, 3, 256]
        x = self.pool(x)   # 输出形状: [8, 3, 1]
        x = x.squeeze(-1)  # 移除长度维度，形状变为 [8, 3]
        x = self.softmax(x)
        # 取概率最大的类别索引，形状变为 [8, 1]
        return torch.argmax(x, dim=1, keepdim=True)

model = ClassificationModel()
input = torch.randn(8, 3, 256)  # 模拟输入数据
output = model(input)            # 输出形状 [8, 1]
print(output.size())