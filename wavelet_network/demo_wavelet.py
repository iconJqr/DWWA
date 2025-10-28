"""
小波网络演示脚本

这个脚本展示了DWWA论文中小波网络的核心创新点：
1. 使用小波变换进行特征下采样
2. 在stem layer融合小波特征
3. 在Bottleneck中使用小波替代stride卷积

运行方式：
    python demo_wavelet.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from wavelet_module import (
    WaveletDownsample,
    WaveletFeatureFusion,
    WaveletStemLayer,
    WaveletBottleneck
)


def visualize_wavelet_decomposition():
    """
    可视化小波分解的效果
    展示如何将图像分解为低频和高频分量
    """
    print("=" * 60)
    print("演示1: 小波分解可视化")
    print("=" * 60)

    # 创建一个简单的测试图像（棋盘格）
    size = 224
    checkerboard = np.zeros((size, size))
    square_size = 28
    for i in range(0, size, square_size):
        for j in range(0, size, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                checkerboard[i:i+square_size, j:j+square_size] = 1

    # 转换为tensor
    x = torch.from_numpy(checkerboard).float().unsqueeze(0).unsqueeze(0)
    print(f"输入图像shape: {x.shape}")

    # 小波变换
    wavelet = WaveletDownsample(wave='db1', mode='zero', J=1)
    coeffs_yl, coeffs_yh = wavelet(x)

    print(f"低频系数(LL) shape: {coeffs_yl.shape}")
    print(f"高频系数 shape: {coeffs_yh[0].shape}")

    # 提取高频分量
    coeffs_lh = coeffs_yh[0][0, 0, 0, :, :].detach().numpy()  # 水平边缘
    coeffs_hl = coeffs_yh[0][0, 0, 1, :, :].detach().numpy()  # 垂直边缘
    coeffs_hh = coeffs_yh[0][0, 0, 2, :, :].detach().numpy()  # 对角边缘
    coeffs_ll = coeffs_yl[0, 0, :, :].detach().numpy()        # 低频近似

    # 可视化（如果matplotlib可用）
    try:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        axes[0, 0].imshow(checkerboard, cmap='gray')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(coeffs_ll, cmap='gray')
        axes[0, 1].set_title('LL - 低频近似\n(主要内容)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(np.abs(coeffs_lh), cmap='hot')
        axes[0, 2].set_title('LH - 水平高频\n(水平边缘)')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(np.abs(coeffs_hl), cmap='hot')
        axes[1, 0].set_title('HL - 垂直高频\n(垂直边缘)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(np.abs(coeffs_hh), cmap='hot')
        axes[1, 1].set_title('HH - 对角高频\n(对角边缘)')
        axes[1, 1].axis('off')

        # 重建图像（所有系数）
        axes[1, 2].text(0.5, 0.5, '小波分解保留了\n所有频率信息\n\n优于传统pooling',
                       ha='center', va='center', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig('wavelet_network/wavelet_decomposition.png', dpi=150, bbox_inches='tight')
        print("✓ 可视化结果已保存到: wavelet_network/wavelet_decomposition.png")
    except Exception as e:
        print(f"警告: 无法生成可视化图像 ({e})")

    print("\n关键优势:")
    print("  - LL保留主要内容（类似下采样结果）")
    print("  - LH/HL/HH保留边缘细节（传统pooling会丢失）")
    print("  - 信息无损，可完全重建")


def compare_downsampling_methods():
    """
    对比不同下采样方法的特征保留能力
    """
    print("\n" + "=" * 60)
    print("演示2: 对比下采样方法")
    print("=" * 60)

    x = torch.randn(1, 64, 56, 56)
    print(f"输入特征: {x.shape}")

    # 方法1: 传统MaxPooling
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    out1 = maxpool(x)
    print(f"\n1. MaxPooling输出: {out1.shape}")
    print(f"   保留信息: {out1.numel()} 个值")
    print(f"   丢失率: {(1 - out1.numel() / x.numel()) * 100:.1f}%")

    # 方法2: Stride=2卷积
    conv_down = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
    out2 = conv_down(x)
    print(f"\n2. Stride卷积输出: {out2.shape}")
    print(f"   保留信息: {out2.numel()} 个值")
    print(f"   丢失率: {(1 - out2.numel() / x.numel()) * 100:.1f}%")

    # 方法3: 小波变换（DWWA方法）
    wavelet = WaveletDownsample()
    coeffs_yl, coeffs_yh = wavelet(x)

    # 计算所有小波系数的总数
    total_wavelet = coeffs_yl.numel() + coeffs_yh[0].numel()

    print(f"\n3. 小波变换输出:")
    print(f"   - LL (低频): {coeffs_yl.shape}")
    print(f"   - LH,HL,HH (高频): {coeffs_yh[0].shape}")
    print(f"   总保留信息: {total_wavelet} 个值")
    print(f"   丢失率: {(1 - total_wavelet / x.numel()) * 100:.1f}%")

    print("\n对比结果:")
    print(f"  MaxPooling:   保留 {out1.numel():6d} 值 (25%)")
    print(f"  Stride卷积:   保留 {out2.numel():6d} 值 (25%)")
    print(f"  小波变换:     保留 {total_wavelet:6d} 值 (100%)")
    print("\n✓ 小波变换是唯一无损的下采样方法!")


def demo_wavelet_stem():
    """
    演示小波Stem层的工作原理
    """
    print("\n" + "=" * 60)
    print("演示3: 小波Stem层")
    print("=" * 60)

    # 模拟RGB图像输入
    x = torch.randn(2, 3, 224, 224)
    print(f"输入图像: {x.shape} (Batch=2, RGB=3, H=224, W=224)")

    # 传统Stem
    traditional_stem = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    # 小波Stem
    wavelet_stem = WaveletStemLayer(in_channels=3, stem_channels=64)

    with torch.no_grad():
        out1 = traditional_stem(x)
        out2 = wavelet_stem(x)

    print(f"\n传统Stem输出: {out1.shape}")
    print(f"小波Stem输出: {out2.shape}")

    print("\n关键区别:")
    print("  传统方法: 7x7卷积 + MaxPool")
    print("            → 只使用卷积学习特征")
    print("\n  小波方法: 7x7卷积 + 小波分解 + 特征融合")
    print("            → 融合卷积特征和小波频域特征")
    print("            → LL(低频) + LH(水平边缘) + HL(垂直边缘) + HH(对角边缘)")

    print("\n优势:")
    print("  ✓ 保留更多细节信息（边缘、纹理）")
    print("  ✓ 多尺度特征表示")
    print("  ✓ 对小目标检测更友好")


def demo_wavelet_bottleneck():
    """
    演示小波Bottleneck的工作原理
    """
    print("\n" + "=" * 60)
    print("演示4: 小波Bottleneck块")
    print("=" * 60)

    x = torch.randn(2, 256, 56, 56)
    print(f"输入特征: {x.shape}")

    # 创建stride=2的bottleneck
    bottleneck = WaveletBottleneck(in_channels=256, out_channels=512, stride=2)

    with torch.no_grad():
        out = bottleneck(x)

    print(f"输出特征: {out.shape}")

    print("\n工作流程:")
    print("  1. 1x1卷积降维: [B,256,56,56] → [B,128,56,56]")
    print("  2. 小波下采样: [B,128,56,56] → [B,128,28,28]")
    print("     ↳ 使用DWT提取低频系数（LL）")
    print("  3. 3x3卷积:    [B,128,28,28] → [B,128,28,28]")
    print("     ↳ stride=1（下采样已由小波完成）")
    print("  4. 1x1卷积升维: [B,128,28,28] → [B,512,28,28]")
    print("  5. 残差连接:   输出 = Conv(输入) + Shortcut")

    print("\n与传统Bottleneck的对比:")
    print("  传统方法: 在3x3卷积使用stride=2")
    print("            → 直接跳过部分像素")
    print("\n  小波方法: 先DWT分解，再在低频上卷积")
    print("            → 保留高频信息用于后续层")


def benchmark_performance():
    """
    性能测试：对比计算成本
    """
    print("\n" + "=" * 60)
    print("演示5: 性能基准测试")
    print("=" * 60)

    x = torch.randn(8, 256, 56, 56).cuda() if torch.cuda.is_available() else torch.randn(8, 256, 56, 56)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print(f"输入: {x.shape}")

    # 传统stride卷积
    conv_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1).to(device)

    # 小波下采样 + 卷积
    wavelet = WaveletDownsample().to(device)
    conv_wavelet = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1).to(device)

    import time

    # 预热
    for _ in range(10):
        _ = conv_down(x)
        _, coeffs = wavelet(x)
        _ = conv_wavelet(coeffs[0] if isinstance(coeffs, tuple) else coeffs)

    # 测试传统方法
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(100):
        _ = conv_down(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.time()
    time_conv = (t2 - t1) / 100

    # 测试小波方法
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(100):
        yl, _ = wavelet(x)
        _ = conv_wavelet(yl)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.time()
    time_wavelet = (t2 - t1) / 100

    print(f"\n传统Stride卷积: {time_conv*1000:.3f} ms")
    print(f"小波+卷积:      {time_wavelet*1000:.3f} ms")
    print(f"额外开销:       {(time_wavelet/time_conv - 1)*100:.1f}%")

    print("\n分析:")
    print("  小波变换增加少量计算开销，但换来:")
    print("  ✓ 更丰富的特征表示")
    print("  ✓ 更好的检测精度")
    print("  ✓ 对小目标更友好")


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  DWWA小波网络 (Wavelet Network) 核心技术演示  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")

    # 演示1: 小波分解可视化
    visualize_wavelet_decomposition()

    # 演示2: 对比下采样方法
    compare_downsampling_methods()

    # 演示3: 小波Stem层
    demo_wavelet_stem()

    # 演示4: 小波Bottleneck
    demo_wavelet_bottleneck()

    # 演示5: 性能测试
    benchmark_performance()

    print("\n" + "=" * 60)
    print("总结: 小波网络的核心创新")
    print("=" * 60)
    print("1. 信息保留: 小波变换无损保留所有频域信息")
    print("2. 多尺度特征: 同时捕获低频内容和高频细节")
    print("3. 边缘增强: LH/HL/HH系数显式编码边缘信息")
    print("4. 即插即用: 可以无缝替换现有网络的下采样层")
    print("5. 检测提升: 特别适合小目标和细节丰富的任务")
    print("=" * 60)

    print("\n✓ 所有演示完成!")
    print("\n文件说明:")
    print("  - wavelet_module.py: 小波网络核心模块")
    print("  - demo_wavelet.py: 演示脚本（本文件）")
    print("  - wavelet_decomposition.png: 小波分解可视化")


if __name__ == '__main__':
    main()
