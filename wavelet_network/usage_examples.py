"""
小波网络使用示例

本文件展示如何在实际项目中使用小波网络模块
"""

# ============================================================================
# 示例1: 构建带小波下采样的ResNet
# ============================================================================

def example1_wavelet_resnet():
    """构建一个使用小波下采样的ResNet"""
    import torch
    import torch.nn as nn
    from wavelet_module import WaveletStemLayer, WaveletBottleneck

    class WaveletResNet(nn.Module):
        """使用小波下采样的ResNet50"""

        def __init__(self, num_classes=1000):
            super(WaveletResNet, self).__init__()

            # 使用小波Stem层
            self.stem = WaveletStemLayer(in_channels=3, stem_channels=64)

            # 构建4个stage
            self.layer1 = self._make_layer(64,   256,  3, stride=1)
            self.layer2 = self._make_layer(256,  512,  4, stride=2)  # 小波下采样
            self.layer3 = self._make_layer(512,  1024, 6, stride=2)  # 小波下采样
            self.layer4 = self._make_layer(1024, 2048, 3, stride=2)  # 小波下采样

            # 分类头
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, num_classes)

        def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            layers = []
            # 第一个block可能有下采样
            layers.append(WaveletBottleneck(in_channels, out_channels, stride))
            # 其余block
            for _ in range(1, num_blocks):
                layers.append(WaveletBottleneck(out_channels, out_channels, stride=1))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    # 测试
    model = WaveletResNet(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    print("示例1: 小波ResNet")
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("✓ 成功!\n")


# ============================================================================
# 示例2: 在现有模型中添加小波下采样
# ============================================================================

def example2_add_wavelet_to_existing_model():
    """在现有模型中添加小波下采样"""
    import torch
    import torch.nn as nn
    from wavelet_module import WaveletDownsample

    class OriginalModel(nn.Module):
        """原始模型（使用MaxPool下采样）"""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool(x)  # 传统下采样
            x = self.conv2(x)
            return x

    class WaveletModel(nn.Module):
        """改进模型（使用小波下采样）"""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.wavelet = WaveletDownsample(wave='db1')  # 替换MaxPool
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x, _ = self.wavelet(x)  # 小波下采样（只用低频）
            x = self.conv2(x)
            return x

    # 对比
    x = torch.randn(1, 3, 224, 224)

    model1 = OriginalModel()
    model2 = WaveletModel()

    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    print("示例2: 替换现有下采样层")
    print(f"原始模型输出: {out1.shape}")
    print(f"小波模型输出: {out2.shape}")
    print("✓ 形状一致，可以无缝替换!\n")


# ============================================================================
# 示例3: 小波多尺度特征提取
# ============================================================================

def example3_multiscale_features():
    """使用小波提取多尺度特征"""
    import torch
    import torch.nn as nn
    from wavelet_module import WaveletDownsample

    class MultiscaleWaveletExtractor(nn.Module):
        """多尺度小波特征提取器"""

        def __init__(self, in_channels):
            super().__init__()
            self.wavelet = WaveletDownsample(wave='db1')

            # 处理不同频率成分的卷积
            self.conv_ll = nn.Conv2d(in_channels, 64, 3, padding=1)  # 低频
            self.conv_lh = nn.Conv2d(in_channels, 64, 3, padding=1)  # 水平高频
            self.conv_hl = nn.Conv2d(in_channels, 64, 3, padding=1)  # 垂直高频
            self.conv_hh = nn.Conv2d(in_channels, 64, 3, padding=1)  # 对角高频

            # 融合层
            self.fusion = nn.Conv2d(256, in_channels, 1)

        def forward(self, x):
            # 小波分解
            coeffs_ll, coeffs_yh = self.wavelet(x)
            coeffs_lh = coeffs_yh[0][:, :, 0, :, :]
            coeffs_hl = coeffs_yh[0][:, :, 1, :, :]
            coeffs_hh = coeffs_yh[0][:, :, 2, :, :]

            # 分别处理每个频率成分
            feat_ll = self.conv_ll(coeffs_ll)
            feat_lh = self.conv_lh(coeffs_lh)
            feat_hl = self.conv_hl(coeffs_hl)
            feat_hh = self.conv_hh(coeffs_hh)

            # 融合所有频率特征
            multiscale_feat = torch.cat([feat_ll, feat_lh, feat_hl, feat_hh], dim=1)
            output = self.fusion(multiscale_feat)

            return output

    # 测试
    model = MultiscaleWaveletExtractor(in_channels=128)
    x = torch.randn(1, 128, 56, 56)
    output = model(x)

    print("示例3: 多尺度小波特征提取")
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    print("特征包含: LL(内容) + LH(水平边缘) + HL(垂直边缘) + HH(对角边缘)")
    print("✓ 成功!\n")


# ============================================================================
# 示例4: 用于目标检测的FPN + 小波
# ============================================================================

def example4_wavelet_fpn():
    """结合小波的FPN特征金字塔"""
    import torch
    import torch.nn as nn
    from wavelet_module import WaveletFeatureFusion

    class WaveletFPN(nn.Module):
        """使用小波增强的FPN"""

        def __init__(self):
            super().__init__()

            # 小波特征融合模块
            self.wavelet_fusion2 = WaveletFeatureFusion(256, 256)
            self.wavelet_fusion3 = WaveletFeatureFusion(512, 256)
            self.wavelet_fusion4 = WaveletFeatureFusion(1024, 256)

            # 侧向连接
            self.lateral2 = nn.Conv2d(256, 256, 1)
            self.lateral3 = nn.Conv2d(256, 256, 1)
            self.lateral4 = nn.Conv2d(256, 256, 1)

            # 平滑层
            self.smooth2 = nn.Conv2d(256, 256, 3, padding=1)
            self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
            self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)

        def _upsample_add(self, x, y):
            """上采样并相加"""
            _, _, H, W = y.size()
            return nn.functional.interpolate(x, size=(H, W), mode='nearest') + y

        def forward(self, c2, c3, c4):
            """
            Args:
                c2: [B, 256,  H/4,  W/4]
                c3: [B, 512,  H/8,  W/8]
                c4: [B, 1024, H/16, W/16]
            """
            # 使用小波融合
            p4 = self.wavelet_fusion4(c4)
            p4 = self.lateral4(p4)

            p3 = self.wavelet_fusion3(c3)
            p3 = self.lateral3(p3)
            p3 = self._upsample_add(p4, p3)

            p2 = self.wavelet_fusion2(c2)
            p2 = self.lateral2(p2)
            p2 = self._upsample_add(p3, p2)

            # 平滑
            p4 = self.smooth4(p4)
            p3 = self.smooth3(p3)
            p2 = self.smooth2(p2)

            return p2, p3, p4

    # 测试
    model = WaveletFPN()
    c2 = torch.randn(1, 256,  56, 56)
    c3 = torch.randn(1, 512,  28, 28)
    c4 = torch.randn(1, 1024, 14, 14)

    p2, p3, p4 = model(c2, c3, c4)

    print("示例4: 小波FPN")
    print(f"C2: {c2.shape} -> P2: {p2.shape}")
    print(f"C3: {c3.shape} -> P3: {p3.shape}")
    print(f"C4: {c4.shape} -> P4: {p4.shape}")
    print("✓ 成功!\n")


# ============================================================================
# 示例5: 训练脚本模板
# ============================================================================

def example5_training_template():
    """训练脚本模板"""
    print("示例5: 训练脚本模板\n")

    training_code = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from wavelet_module import WaveletBottleneck, WaveletStemLayer

# 1. 定义模型
class MyWaveletNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = WaveletStemLayer(3, 64)
        self.layer1 = WaveletBottleneck(64, 256, stride=1)
        self.layer2 = WaveletBottleneck(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 2. 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyWaveletNet(num_classes=1000).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练循环
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 4. 运行训练
# for epoch in range(num_epochs):
#     loss = train_epoch(model, train_loader, criterion, optimizer, device)
#     print(f'Epoch {epoch}: Loss = {loss:.4f}')
"""

    print(training_code)
    print("✓ 训练模板已生成!\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("小波网络使用示例".center(70))
    print("=" * 70 + "\n")

    try:
        example1_wavelet_resnet()
    except ImportError as e:
        print(f"示例1跳过: {e}\n")

    try:
        example2_add_wavelet_to_existing_model()
    except ImportError as e:
        print(f"示例2跳过: {e}\n")

    try:
        example3_multiscale_features()
    except ImportError as e:
        print(f"示例3跳过: {e}\n")

    try:
        example4_wavelet_fpn()
    except ImportError as e:
        print(f"示例4跳过: {e}\n")

    example5_training_template()

    print("=" * 70)
    print("所有示例运行完成!".center(70))
    print("=" * 70)
    print("\n提示:")
    print("  1. 这些示例展示了小波网络在不同场景的应用")
    print("  2. 可以直接复制代码用于您的项目")
    print("  3. 小波下采样可以无缝替换现有的MaxPool/Stride卷积")
    print("  4. 对于小目标检测任务，推荐使用小波FPN")
    print()


if __name__ == '__main__':
    main()
