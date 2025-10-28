"""
最简单的小波变换示例

不需要任何库，纯Python实现，帮助理解计算过程
"""

import math


def simple_dwt_example():
    """
    最简单的1D小波变换示例
    """
    print("=" * 70)
    print("示例1: 一维小波变换 (最简单)")
    print("=" * 70)

    # 输入信号
    signal = [1, 2, 3, 4]
    print(f"\n输入信号: {signal}")

    # db1(Haar)小波滤波器
    sqrt2 = math.sqrt(2)
    h = [1/sqrt2, 1/sqrt2]      # 低通: 平均
    g = [1/sqrt2, -1/sqrt2]     # 高通: 差分

    print(f"\n滤波器系数:")
    print(f"  低通(h): [{h[0]:.3f}, {h[1]:.3f}]  ← 计算平均值")
    print(f"  高通(g): [{g[0]:.3f}, {g[1]:.3f}]  ← 计算差值")

    # 计算低频系数
    print(f"\n计算低频系数 (近似):")
    low0 = h[0] * signal[0] + h[1] * signal[1]
    low1 = h[0] * signal[2] + h[1] * signal[3]
    print(f"  low[0] = {h[0]:.3f} × {signal[0]} + {h[1]:.3f} × {signal[1]} = {low0:.3f}")
    print(f"  low[1] = {h[0]:.3f} × {signal[2]} + {h[1]:.3f} × {signal[3]} = {low1:.3f}")
    print(f"  → 低频: [{low0:.3f}, {low1:.3f}]")

    # 计算高频系数
    print(f"\n计算高频系数 (细节):")
    high0 = g[0] * signal[0] + g[1] * signal[1]
    high1 = g[0] * signal[2] + g[1] * signal[3]
    print(f"  high[0] = {g[0]:.3f} × {signal[0]} + {g[1]:.3f} × {signal[1]} = {high0:.3f}")
    print(f"  high[1] = {g[0]:.3f} × {signal[2]} + {g[1]:.3f} × {signal[3]} = {high1:.3f}")
    print(f"  → 高频: [{high0:.3f}, {high1:.3f}]")

    print(f"\n结果:")
    print(f"  输入: {signal} (4个值)")
    print(f"  输出: 低频{[low0, low1]} + 高频{[high0, high1]} (4个值)")
    print(f"  ✓ 信息量相同，但分成了'粗略'和'细节'两部分")


def simple_2d_dwt_example():
    """
    简单的2D小波变换示例
    """
    print("\n" + "=" * 70)
    print("示例2: 二维小波变换 (图像)")
    print("=" * 70)

    # 4×4图像
    image = [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4]
    ]

    print("\n输入图像 (4×4):")
    for row in image:
        print("  ", row)

    sqrt2 = math.sqrt(2)

    # 步骤1: 对每行做1D-DWT
    print("\n步骤1: 对每行做小波变换")
    print("-" * 70)

    row_results = []
    for i, row in enumerate(image):
        # 低频 = (a+b)/√2
        low0 = (row[0] + row[1]) / sqrt2
        low1 = (row[2] + row[3]) / sqrt2

        # 高频 = (a-b)/√2
        high0 = (row[0] - row[1]) / sqrt2
        high1 = (row[2] - row[3]) / sqrt2

        print(f"  行{i}: {row}")
        print(f"    低频: [{low0:.2f}, {low1:.2f}]")
        print(f"    高频: [{high0:.2f}, {high1:.2f}]")

        row_results.append({
            'low': [low0, low1],
            'high': [high0, high1]
        })

    # 提取L和H矩阵
    L_matrix = [[r['low'][0], r['low'][1]] for r in row_results]
    H_matrix = [[r['high'][0], r['high'][1]] for r in row_results]

    print(f"\n  L矩阵 (低频部分):")
    for row in L_matrix:
        print(f"    [{row[0]:5.2f}, {row[1]:5.2f}]")

    print(f"\n  H矩阵 (高频部分):")
    for row in H_matrix:
        print(f"    [{row[0]:5.2f}, {row[1]:5.2f}]")

    # 步骤2: 对列做1D-DWT
    print("\n步骤2: 对L和H的每列做小波变换")
    print("-" * 70)

    # 对L矩阵的列
    print("\n  处理L矩阵:")
    LL = []
    HL = []
    for col in range(2):
        col_data = [L_matrix[row][col] for row in range(4)]
        print(f"    列{col}: {[f'{x:.2f}' for x in col_data]}")

        # 低频
        ll0 = (col_data[0] + col_data[1]) / sqrt2
        ll1 = (col_data[2] + col_data[3]) / sqrt2
        print(f"      → LL: [{ll0:.2f}, {ll1:.2f}]")

        # 高频
        hl0 = (col_data[0] - col_data[1]) / sqrt2
        hl1 = (col_data[2] - col_data[3]) / sqrt2
        print(f"      → HL: [{hl0:.2f}, {hl1:.2f}]")

        LL.append([ll0, ll1])
        HL.append([hl0, hl1])

    # 对H矩阵的列
    print("\n  处理H矩阵:")
    LH = []
    HH = []
    for col in range(2):
        col_data = [H_matrix[row][col] for row in range(4)]
        print(f"    列{col}: {[f'{x:.2f}' for x in col_data]}")

        # 低频
        lh0 = (col_data[0] + col_data[1]) / sqrt2
        lh1 = (col_data[2] + col_data[3]) / sqrt2
        print(f"      → LH: [{lh0:.2f}, {lh1:.2f}]")

        # 高频
        hh0 = (col_data[0] - col_data[1]) / sqrt2
        hh1 = (col_data[2] - col_data[3]) / sqrt2
        print(f"      → HH: [{hh0:.2f}, {hh1:.2f}]")

        LH.append([lh0, lh1])
        HH.append([hh0, hh1])

    # 转置以正确显示
    LL_t = [[LL[0][0], LL[1][0]], [LL[0][1], LL[1][1]]]
    HL_t = [[HL[0][0], HL[1][0]], [HL[0][1], HL[1][1]]]
    LH_t = [[LH[0][0], LH[1][0]], [LH[0][1], LH[1][1]]]
    HH_t = [[HH[0][0], HH[1][0]], [HH[0][1], HH[1][1]]]

    # 显示最终结果
    print("\n" + "=" * 70)
    print("最终结果: 四个子带")
    print("=" * 70)

    print("\nLL (低频低频) - 图像缩略版:")
    for row in LL_t:
        print(f"  [{row[0]:5.2f}, {row[1]:5.2f}]")

    print("\nLH (低频高频) - 水平边缘:")
    for row in LH_t:
        print(f"  [{row[0]:5.2f}, {row[1]:5.2f}]")

    print("\nHL (高频低频) - 垂直边缘:")
    for row in HL_t:
        print(f"  [{row[0]:5.2f}, {row[1]:5.2f}]")

    print("\nHH (高频高频) - 对角边缘:")
    for row in HH_t:
        print(f"  [{row[0]:5.2f}, {row[1]:5.2f}]")

    # 统计
    print("\n信息统计:")
    print(f"  原始图像: 4×4 = 16 个值")
    print(f"  小波系数: LL(4) + LH(4) + HL(4) + HH(4) = 16 个值")
    print(f"  ✓ 信息完全保留!")


def explain_physical_meaning():
    """
    解释四个子带的物理意义
    """
    print("\n" + "=" * 70)
    print("示例3: 四个子带的物理意义")
    print("=" * 70)

    examples = [
        ("纯色块", [
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5]
        ]),
        ("水平边缘", [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [9, 9, 9, 9],
            [9, 9, 9, 9]
        ]),
        ("垂直边缘", [
            [1, 1, 9, 9],
            [1, 1, 9, 9],
            [1, 1, 9, 9],
            [1, 1, 9, 9]
        ]),
        ("棋盘格", [
            [1, 9, 1, 9],
            [9, 1, 9, 1],
            [1, 9, 1, 9],
            [9, 1, 9, 1]
        ])
    ]

    sqrt2 = math.sqrt(2)

    for name, image in examples:
        print(f"\n{name}:")
        for row in image:
            print(f"  {row}")

        # 简单计算一个代表性的系数
        # LL: 左上角2×2的平均
        ll = sum(image[0][:2] + image[1][:2]) / 4

        # LH: 上下差异 (行0-行1)
        lh = abs(image[0][0] - image[1][0])

        # HL: 左右差异 (列0-列1)
        hl = abs(image[0][0] - image[0][1])

        # HH: 对角差异
        hh = abs(image[0][0] - image[1][1])

        print(f"  → LL≈{ll:.1f} (整体亮度)")
        print(f"  → LH≈{lh:.1f} (水平变化)")
        print(f"  → HL≈{hl:.1f} (垂直变化)")
        print(f"  → HH≈{hh:.1f} (对角变化)")


def compare_with_maxpool():
    """
    对比小波变换和MaxPooling
    """
    print("\n" + "=" * 70)
    print("对比: 小波变换 vs MaxPooling")
    print("=" * 70)

    image = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    print("\n原始图像 (4×4):")
    for row in image:
        print(f"  {row}")

    # MaxPooling
    print("\nMaxPooling (2×2):")
    pool_result = [
        [max(image[0][0], image[0][1], image[1][0], image[1][1]),
         max(image[0][2], image[0][3], image[1][2], image[1][3])],
        [max(image[2][0], image[2][1], image[3][0], image[3][1]),
         max(image[2][2], image[2][3], image[3][2], image[3][3])]
    ]

    for row in pool_result:
        print(f"  {row}")

    print(f"\n  保留信息: {[item for row in pool_result for item in row]}")
    print(f"  丢失信息: 原16个值 → 保留4个值 → 丢失12个值(75%)")

    # 小波变换
    print("\n小波变换:")
    print("  LL: [主要内容]  2×2 = 4个值")
    print("  LH: [水平边缘]  2×2 = 4个值")
    print("  HL: [垂直边缘]  2×2 = 4个值")
    print("  HH: [对角边缘]  2×2 = 4个值")
    print(f"\n  保留信息: 4+4+4+4 = 16个值")
    print(f"  丢失信息: 0个值(0%)")

    print("\n结论:")
    print("  • MaxPooling: 只保留最大值，丢失位置和其他值")
    print("  • 小波变换: 保留所有信息，分为低频和高频")
    print("  • 小波可以完美重建原图，MaxPooling不可以")


def show_code_usage():
    """
    展示在实际代码中如何使用
    """
    print("\n" + "=" * 70)
    print("在代码中的使用")
    print("=" * 70)

    code = '''
# 方法1: 使用pytorch_wavelets库
from pytorch_wavelets import DWTForward
import torch

# 初始化
xfm = DWTForward(J=1, wave='db1', mode='zero')

# 输入图像 [Batch, Channel, Height, Width]
x = torch.randn(2, 3, 224, 224)

# 小波分解
yl, yh = xfm(x)

# yl: [2, 3, 112, 112]       低频(LL)
# yh: [2, 3, 3, 112, 112]    高频(LH, HL, HH)

# 提取高频分量
lh = yh[0][:, :, 0, :, :]  # 水平
hl = yh[0][:, :, 1, :, :]  # 垂直
hh = yh[0][:, :, 2, :, :]  # 对角

# 拼接所有系数
all_coeffs = torch.cat([yl, lh, hl, hh], dim=1)
# [2, 12, 112, 112]

# 方法2: 使用我们的WaveletDownsample模块
from wavelet_module import WaveletDownsample

wavelet = WaveletDownsample(wave='db1')
yl, yh = wavelet(x)
'''

    print(code)


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "小波变换简单示例 - 不需要任何库".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # 1. 最简单的1D示例
    simple_dwt_example()

    # 2. 2D图像示例
    simple_2d_dwt_example()

    # 3. 物理意义
    explain_physical_meaning()

    # 4. 对比MaxPooling
    compare_with_maxpool()

    # 5. 代码使用
    show_code_usage()

    print("\n" + "=" * 70)
    print("核心要点")
    print("=" * 70)
    print("""
    1. 小波变换的本质
       • 把信号分成"粗略"(低频)和"细节"(高频)
       • 低频 = 相邻值的平均 (保留主要内容)
       • 高频 = 相邻值的差异 (保留边缘细节)

    2. 计算公式 (db1小波)
       • 低频 = (a + b) / √2
       • 高频 = (a - b) / √2
       • 就这么简单!

    3. 2D小波 = 两次1D小波
       • 先对每行变换 → 得到L和H
       • 再对L和H的每列变换 → 得到LL,HL,LH,HH

    4. 为什么比MaxPooling好
       • MaxPooling: 丢弃75%信息
       • 小波: 保留100%信息
       • 小波可以完美重建，MaxPooling不可以

    5. 在DWWA代码中的作用
       • Stem层: 融合小波特征到网络输入
       • Bottleneck: 用小波替代stride=2卷积
       • 效果: 保留细节，提升小目标检测精度
    """)

    print("\n✓ 所有示例完成!")
    print("\n提示:")
    print("  - 这个脚本不需要任何库，可以直接运行")
    print("  - 运行 visualize_computation.py 查看更详细的可视化")
    print("  - 阅读 WAVELET_PRINCIPLE.md 了解完整的数学原理")


if __name__ == '__main__':
    main()
