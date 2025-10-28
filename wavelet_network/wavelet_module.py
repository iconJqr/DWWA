"""
Wavelet Network Module - 小波网络核心模块

这个模块实现了使用离散小波变换（DWT）替代传统下采样的创新方法。
核心创新点：在降采样的同时保留更多高频细节信息。

主要功能：
1. 使用DWT进行特征分解，获取低频和高频系数
2. 将小波系数与原始特征融合
3. 在下采样时保留更丰富的特征信息
"""

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


class WaveletDownsample(nn.Module):
    """
    小波下采样模块

    使用离散小波变换进行下采样，相比传统的stride卷积或pooling，
    能够保留更多的高频细节信息。

    Args:
        wave (str): 小波基，默认'db1' (Daubechies 1)
        mode (str): 边界处理模式，默认'zero'
        J (int): 分解层数，默认1
    """

    def __init__(self, wave='db1', mode='zero', J=1):
        super(WaveletDownsample, self).__init__()
        self.xfm = DWTForward(J=J, wave=wave, mode=mode)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            coeffs_yl: 低频系数 [B, C, H/2, W/2]
            coeffs_yh: 高频系数列表，包含3个方向 (LH, HL, HH)
        """
        coeffs_yl, coeffs_yh = self.xfm(x)
        return coeffs_yl, coeffs_yh


class WaveletFeatureFusion(nn.Module):
    """
    小波特征融合模块

    将小波变换得到的低频和高频系数进行融合，
    生成包含更丰富信息的特征表示。

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        wave (str): 小波基
    """

    def __init__(self, in_channels, out_channels, wave='db1'):
        super(WaveletFeatureFusion, self).__init__()
        self.xfm = DWTForward(J=1, wave=wave, mode='zero')

        # 小波分解后通道数变化：
        # LL: in_channels
        # LH, HL, HH: 各in_channels
        # 总共: in_channels * 4
        wavelet_channels = in_channels * 4

        # 融合卷积：将小波系数融合到指定输出通道数
        self.fusion_conv = nn.Conv2d(
            wavelet_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            融合后的特征 [B, out_channels, H/2, W/2]
        """
        # 小波分解
        coeffs_yl, coeffs_yh = self.xfm(x)

        # 提取三个高频方向的系数
        # coeffs_yh shape: [B, C, 3, H/2, W/2]
        coeffs_lh = coeffs_yh[0][:, :, 0, :, :]  # LH (水平高频)
        coeffs_hl = coeffs_yh[0][:, :, 1, :, :]  # HL (垂直高频)
        coeffs_hh = coeffs_yh[0][:, :, 2, :, :]  # HH (对角高频)

        # 拼接所有系数
        # LL: 低频近似，包含主要信息
        # LH: 水平边缘信息
        # HL: 垂直边缘信息
        # HH: 对角边缘信息
        wavelet_features = torch.cat([
            coeffs_yl,    # 低频
            coeffs_lh,    # 水平高频
            coeffs_hl,    # 垂直高频
            coeffs_hh     # 对角高频
        ], dim=1)

        # 通过1x1卷积融合
        fused_features = self.fusion_conv(wavelet_features)

        return fused_features


class WaveletStemLayer(nn.Module):
    """
    小波Stem层

    在网络的输入stem阶段使用小波变换，将原始图像特征与小波系数融合。
    这是DWWA论文中的关键创新之一。

    Args:
        in_channels (int): 输入通道数（通常为3，RGB图像）
        stem_channels (int): stem输出通道数（通常为64）
    """

    def __init__(self, in_channels=3, stem_channels=64):
        super(WaveletStemLayer, self).__init__()

        # 传统7x7卷积
        self.conv1 = nn.Conv2d(
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(stem_channels)
        self.relu = nn.ReLU(inplace=True)

        # 小波变换
        self.xfm = DWTForward(J=1, wave='db1', mode='zero')

        # 小波特征通道数：3(RGB) * 4(LL,LH,HL,HH) = 12
        wavelet_channels = in_channels * 4

        # 融合卷积：将stem特征(64) + 小波特征(12) -> stem_channels(64)
        self.fusion_conv = nn.Conv2d(
            stem_channels + wavelet_channels,
            stem_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            融合了小波信息的stem特征 [B, stem_channels, H/4, W/4]
        """
        # 小波分解原始图像
        coeffs, coeffs_ = self.xfm(x)
        coeffs_lh = coeffs_[0][:, :, 0, :, :]
        coeffs_hl = coeffs_[0][:, :, 1, :, :]
        coeffs_hh = coeffs_[0][:, :, 2, :, :]

        # 拼接所有小波系数
        wavelet_features = torch.cat([
            coeffs,       # LL: 低频近似
            coeffs_lh,    # LH: 水平边缘
            coeffs_hl,    # HL: 垂直边缘
            coeffs_hh     # HH: 对角边缘
        ], dim=1)

        # 传统卷积路径
        x = self.conv1(x)

        # 融合小波特征和卷积特征
        x = torch.cat([x, wavelet_features], dim=1)
        x = self.fusion_conv(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class WaveletBottleneck(nn.Module):
    """
    小波Bottleneck块

    在ResNet Bottleneck的下采样阶段使用小波变换替代stride=2的卷积。
    这是DWWA的核心创新：在降采样时保留更多特征信息。

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        stride (int): 步长，2表示下采样
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(WaveletBottleneck, self).__init__()

        mid_channels = out_channels // 4

        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3 conv with wavelet downsampling
        self.stride = stride
        if stride == 2:
            # 使用小波变换进行下采样
            self.xfm = DWTForward(J=1, wave='db1', mode='zero')
            self.conv2 = nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,  # 注意：这里stride=1，下采样由小波完成
                padding=1,
                bias=False
            )
        else:
            self.conv2 = nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )

        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 conv
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [B, in_channels, H, W]

        Returns:
            输出特征 [B, out_channels, H/stride, W/stride]
        """
        identity = x

        # Conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Conv2 with wavelet downsampling
        if self.stride == 2:
            # 关键创新：使用小波变换降采样
            coeffs_yl, _ = self.xfm(out)  # 只使用低频系数
            out = self.conv2(coeffs_yl)
        else:
            out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        # Conv3
        out = self.conv3(out)
        out = self.bn3(out)

        # Shortcut
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


if __name__ == '__main__':
    # 测试代码
    print("Testing Wavelet Network Modules...")

    # 测试WaveletDownsample
    print("\n1. Testing WaveletDownsample:")
    x = torch.randn(2, 64, 56, 56)
    wavelet_down = WaveletDownsample()
    yl, yh = wavelet_down(x)
    print(f"Input shape: {x.shape}")
    print(f"Low-freq (LL) shape: {yl.shape}")
    print(f"High-freq shape: {yh[0].shape}")

    # 测试WaveletFeatureFusion
    print("\n2. Testing WaveletFeatureFusion:")
    x = torch.randn(2, 64, 56, 56)
    fusion = WaveletFeatureFusion(64, 128)
    out = fusion(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # 测试WaveletStemLayer
    print("\n3. Testing WaveletStemLayer:")
    x = torch.randn(2, 3, 224, 224)
    stem = WaveletStemLayer(3, 64)
    out = stem(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # 测试WaveletBottleneck
    print("\n4. Testing WaveletBottleneck:")
    x = torch.randn(2, 256, 56, 56)
    bottleneck = WaveletBottleneck(256, 512, stride=2)
    out = bottleneck(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    print("\nAll tests passed!")
