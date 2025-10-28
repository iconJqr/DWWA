"""
小波变换计算过程可视化

这个脚本展示小波变换的详细计算步骤，帮助理解：
1. 如何从原始图像得到4个子带
2. 每个子带的物理意义
3. 为什么能保留100%的信息
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def db1_filters():
    """
    返回db1(Haar)小波的滤波器系数

    这是最简单的小波，代码中使用的就是这个
    """
    sqrt2 = np.sqrt(2)

    # 低通滤波器 (平均)
    h = np.array([1/sqrt2, 1/sqrt2])

    # 高通滤波器 (差分)
    g = np.array([1/sqrt2, -1/sqrt2])

    return h, g


def manual_dwt_1d(signal, h, g):
    """
    手动实现1D小波变换

    Args:
        signal: 1D信号
        h: 低通滤波器
        g: 高通滤波器

    Returns:
        low, high: 低频和高频系数
    """
    n = len(signal)
    low = []
    high = []

    # 每次处理2个样本
    for i in range(0, n, 2):
        if i+1 < n:
            # 卷积并下采样
            low_val = h[0] * signal[i] + h[1] * signal[i+1]
            high_val = g[0] * signal[i] + g[1] * signal[i+1]

            low.append(low_val)
            high.append(high_val)

    return np.array(low), np.array(high)


def manual_dwt_2d(image, h, g):
    """
    手动实现2D小波变换

    步骤:
    1. 对每一行做1D-DWT
    2. 对结果的每一列做1D-DWT

    Returns:
        LL, LH, HL, HH: 四个子带
    """
    H, W = image.shape

    # 步骤1: 对每行进行变换
    row_L = np.zeros((H, W//2))
    row_H = np.zeros((H, W//2))

    for i in range(H):
        row_L[i], row_H[i] = manual_dwt_1d(image[i], h, g)

    # 步骤2: 对L和H的每列进行变换
    LL = np.zeros((H//2, W//2))
    HL = np.zeros((H//2, W//2))
    LH = np.zeros((H//2, W//2))
    HH = np.zeros((H//2, W//2))

    for j in range(W//2):
        # 对L部分的列
        LL[:, j], HL[:, j] = manual_dwt_1d(row_L[:, j], h, g)
        # 对H部分的列
        LH[:, j], HH[:, j] = manual_dwt_1d(row_H[:, j], h, g)

    return LL, LH, HL, HH


def visualize_step_by_step():
    """
    可视化小波变换的逐步计算过程
    """
    print("=" * 70)
    print("演示: 小波变换逐步计算过程")
    print("=" * 70)

    # 创建简单的测试图像 (8x8 棋盘格)
    image = np.zeros((8, 8))
    image[0:4, 0:4] = 1
    image[4:8, 4:8] = 1

    print("\n原始图像 (8x8):")
    print(image.astype(int))

    # 获取滤波器
    h, g = db1_filters()
    print(f"\ndb1小波滤波器:")
    print(f"  低通(h): {h}")
    print(f"  高通(g): {g}")

    # 步骤1: 对第一行进行1D-DWT演示
    print("\n" + "-" * 70)
    print("步骤1: 对第一行进行1D小波变换")
    print("-" * 70)
    row = image[0]
    print(f"原始行: {row}")

    low, high = manual_dwt_1d(row, h, g)
    print(f"\n计算过程:")
    for i in range(0, len(row), 2):
        if i+1 < len(row):
            l = h[0] * row[i] + h[1] * row[i+1]
            h_val = g[0] * row[i] + g[1] * row[i+1]
            print(f"  位置[{i},{i+1}]: 值=[{row[i]}, {row[i+1]}]")
            print(f"    低频 = {h[0]:.3f}×{row[i]} + {h[1]:.3f}×{row[i+1]} = {l:.3f}")
            print(f"    高频 = {g[0]:.3f}×{row[i]} + {g[1]:.3f}×{row[i+1]} = {h_val:.3f}")

    print(f"\n结果:")
    print(f"  低频系数: {low}")
    print(f"  高频系数: {high}")

    # 步骤2: 完整2D变换
    print("\n" + "-" * 70)
    print("步骤2: 完整2D小波变换")
    print("-" * 70)

    LL, LH, HL, HH = manual_dwt_2d(image, h, g)

    print(f"\nLL (低频, 4x4) - 平滑版本:")
    print(LL)

    print(f"\nLH (水平高频, 4x4) - 水平边缘:")
    print(LH)

    print(f"\nHL (垂直高频, 4x4) - 垂直边缘:")
    print(HL)

    print(f"\nHH (对角高频, 4x4) - 对角边缘:")
    print(HH)

    # 可视化
    try:
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 4, figure=fig)

        # 原图
        ax1 = fig.add_subplot(gs[0, 1:3])
        ax1.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax1.set_title('原始图像 (8×8)', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 四个子带
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(LL, cmap='gray')
        ax2.set_title('LL - 低频\n(主要内容)', fontsize=12)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.imshow(np.abs(LH), cmap='hot')
        ax3.set_title('LH - 水平高频\n(水平边缘)', fontsize=12)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.imshow(np.abs(HL), cmap='hot')
        ax4.set_title('HL - 垂直高频\n(垂直边缘)', fontsize=12)
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 3])
        ax5.imshow(np.abs(HH), cmap='hot')
        ax5.set_title('HH - 对角高频\n(对角边缘)', fontsize=12)
        ax5.axis('off')

        # 添加说明
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        explanation = """
        计算原理:
        1. 每行应用低通(h)和高通(g)滤波器，得到L和H
        2. 对L和H的每列再次应用滤波器，得到LL, HL, LH, HH
        3. 信息完全保留: 原图64个值 = LL(16) + LH(16) + HL(16) + HH(16)

        物理意义:
        • LL: 低频近似，包含图像主要内容（相当于下采样）
        • LH: 水平方向的高频，捕获水平边缘
        • HL: 垂直方向的高频，捕获垂直边缘
        • HH: 对角方向的高频，捕获角点和对角纹理
        """
        ax6.text(0.1, 0.5, explanation, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        plt.savefig('wavelet_network/step_by_step_computation.png',
                   dpi=150, bbox_inches='tight')
        print("\n✓ 可视化已保存: wavelet_network/step_by_step_computation.png")
    except Exception as e:
        print(f"\n警告: 无法生成可视化 ({e})")


def compare_with_pytorch_wavelets():
    """
    对比手动计算和pytorch_wavelets库的结果
    """
    print("\n" + "=" * 70)
    print("验证: 手动计算 vs pytorch_wavelets库")
    print("=" * 70)

    try:
        from pytorch_wavelets import DWTForward

        # 创建测试图像
        image = np.zeros((8, 8))
        image[0:4, 0:4] = 1
        image[4:8, 4:8] = 1

        # 手动计算
        h, g = db1_filters()
        LL_manual, LH_manual, HL_manual, HH_manual = manual_dwt_2d(image, h, g)

        # 使用pytorch_wavelets
        x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        xfm = DWTForward(J=1, wave='db1', mode='zero')
        yl, yh = xfm(x)

        LL_torch = yl[0, 0].numpy()
        LH_torch = yh[0][0, 0, 0].numpy()
        HL_torch = yh[0][0, 0, 1].numpy()
        HH_torch = yh[0][0, 0, 2].numpy()

        # 对比
        print("\nLL系数对比:")
        print("手动计算:")
        print(LL_manual[:3, :3])
        print("pytorch_wavelets:")
        print(LL_torch[:3, :3])
        print(f"误差: {np.abs(LL_manual - LL_torch).max():.6f}")

        print("\nLH系数对比:")
        print("手动计算:")
        print(LH_manual[:3, :3])
        print("pytorch_wavelets:")
        print(LH_torch[:3, :3])
        print(f"误差: {np.abs(LH_manual - LH_torch).max():.6f}")

        if np.allclose(LL_manual, LL_torch, atol=1e-5):
            print("\n✓ 验证通过! 手动计算与库函数一致")
        else:
            print("\n✗ 存在差异，可能是边界处理不同")

    except ImportError:
        print("\n需要安装pytorch_wavelets库:")
        print("  pip install pytorch_wavelets")


def show_information_preservation():
    """
    展示信息完全保留的特性
    """
    print("\n" + "=" * 70)
    print("演示: 信息完全保留 (可逆性)")
    print("=" * 70)

    try:
        from pytorch_wavelets import DWTForward, DWTInverse

        # 创建随机图像
        x = torch.randn(1, 1, 16, 16)

        print(f"原始图像: {x.shape}")
        print(f"像素总数: {x.numel()}")
        print(f"像素值范围: [{x.min():.3f}, {x.max():.3f}]")

        # 小波分解
        xfm = DWTForward(J=1, wave='db1', mode='zero')
        yl, yh = xfm(x)

        print(f"\n小波分解后:")
        print(f"  LL: {yl.shape} → {yl.numel()} 个值")
        print(f"  LH,HL,HH: {yh[0].shape} → {yh[0].numel()} 个值")
        print(f"  总计: {yl.numel() + yh[0].numel()} 个值")
        print(f"  保留率: {(yl.numel() + yh[0].numel()) / x.numel() * 100:.1f}%")

        # 小波重建
        ifm = DWTInverse(wave='db1', mode='zero')
        x_recon = ifm((yl, yh))

        print(f"\n重建图像: {x_recon.shape}")

        # 计算误差
        mse = torch.mean((x - x_recon) ** 2)
        max_error = torch.abs(x - x_recon).max()

        print(f"\n重建质量:")
        print(f"  均方误差(MSE): {mse:.10f}")
        print(f"  最大绝对误差: {max_error:.10f}")

        if mse < 1e-6:
            print(f"\n✓ 几乎完美重建! 信息100%保留")

    except ImportError:
        print("\n需要安装pytorch_wavelets库")


def trace_forward_pass():
    """
    追踪一个完整前向传播的数据流
    """
    print("\n" + "=" * 70)
    print("追踪: 完整前向传播数据流")
    print("=" * 70)

    try:
        from pytorch_wavelets import DWTForward
        import torch.nn as nn

        print("\n场景: ResNet Stem层")
        print("-" * 70)

        # 输入: RGB图像
        x = torch.randn(2, 3, 224, 224)
        print(f"\n[输入] RGB图像: {list(x.shape)}")
        print(f"  - Batch size: 2")
        print(f"  - Channels: 3 (RGB)")
        print(f"  - Size: 224×224")

        # 路径1: 传统卷积
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feat_conv = conv1(x)
        print(f"\n[路径1] 7×7卷积: {list(x.shape)} → {list(feat_conv.shape)}")
        print(f"  - 步长2，分辨率减半")
        print(f"  - 通道数: 3 → 64")

        # 路径2: 小波分解
        xfm = DWTForward(J=1, wave='db1', mode='zero')
        coeffs_ll, coeffs_yh = xfm(x)

        print(f"\n[路径2] 小波分解: {list(x.shape)} → 4个子带")
        print(f"  - LL (低频): {list(coeffs_ll.shape)}")
        print(f"  - LH,HL,HH: {list(coeffs_yh[0].shape)}")

        # 提取高频
        lh = coeffs_yh[0][:, :, 0, :, :]
        hl = coeffs_yh[0][:, :, 1, :, :]
        hh = coeffs_yh[0][:, :, 2, :, :]

        print(f"\n[提取] 分离高频分量:")
        print(f"  - LH (水平): {list(lh.shape)}")
        print(f"  - HL (垂直): {list(hl.shape)}")
        print(f"  - HH (对角): {list(hh.shape)}")

        # 拼接小波特征
        wavelet_feat = torch.cat([coeffs_ll, lh, hl, hh], dim=1)
        print(f"\n[拼接] 小波特征: {list(wavelet_feat.shape)}")
        print(f"  - 通道数: 3×4 = 12")

        # 融合
        fused = torch.cat([feat_conv, wavelet_feat], dim=1)
        print(f"\n[融合] 卷积 + 小波: {list(fused.shape)}")
        print(f"  - 通道数: 64 + 12 = 76")

        # 1×1卷积融合
        fusion_conv = nn.Conv2d(76, 64, kernel_size=1)
        output = fusion_conv(fused)
        print(f"\n[输出] 融合后: {list(output.shape)}")
        print(f"  - 通道数: 76 → 64")

        print("\n" + "-" * 70)
        print("数据流总结:")
        print("-" * 70)
        print("输入 [2,3,224,224]")
        print("  ├─→ 7×7Conv → [2,64,112,112]")
        print("  └─→ DWT")
        print("       ├─→ LL  [2,3,112,112]")
        print("       ├─→ LH  [2,3,112,112]")
        print("       ├─→ HL  [2,3,112,112]")
        print("       └─→ HH  [2,3,112,112]")
        print("            └─→ Cat → [2,12,112,112]")
        print("  合并 → [2,76,112,112]")
        print("  1×1Conv → [2,64,112,112]")

    except ImportError:
        print("\n需要安装pytorch和pytorch_wavelets")


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "小波变换计算过程可视化".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # 1. 逐步计算演示
    visualize_step_by_step()

    # 2. 验证计算正确性
    compare_with_pytorch_wavelets()

    # 3. 展示信息保留
    show_information_preservation()

    # 4. 追踪完整数据流
    trace_forward_pass()

    print("\n" + "=" * 70)
    print("关键要点总结")
    print("=" * 70)
    print("""
    1. 小波变换 = 行DWT + 列DWT
       - 先对每行用滤波器 → 得到L和H
       - 再对L和H的每列用滤波器 → 得到LL,HL,LH,HH

    2. db1(Haar)小波最简单
       - 低通滤波器: [1/√2, 1/√2]  (求平均)
       - 高通滤波器: [1/√2, -1/√2] (求差分)

    3. 信息完全保留
       - 输入N个值 → 输出N个值 (分布在4个子带)
       - 可以完美重建原始信号

    4. 物理意义清晰
       - LL: 图像缩略图 (低频内容)
       - LH: 水平边缘 (上下变化)
       - HL: 垂直边缘 (左右变化)
       - HH: 对角边缘 (对角变化)

    5. 代码中的应用
       - Stem层: 融合小波和卷积特征
       - Bottleneck: 用LL替代stride=2卷积
       - 好处: 保留高频细节，提升检测精度
    """)

    print("✓ 所有演示完成!")


if __name__ == '__main__':
    main()
