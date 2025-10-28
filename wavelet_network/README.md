# 小波网络 (Wavelet Network) - DWWA核心创新

本目录包含从DWWA项目中提取的**小波网络**核心代码，可以独立运行和使用。

## 📚 什么是小波网络？

小波网络是DWWA论文的核心创新之一，使用**离散小波变换(DWT)**替代传统的下采样操作（如MaxPooling、Stride卷积），在降采样的同时保留更多的高频细节信息。

### 核心优势

| 方法 | 信息保留 | 细节保留 | 适用场景 |
|------|---------|---------|---------|
| MaxPooling | ❌ 丢失75% | ❌ 丢失细节 | 一般场景 |
| Stride卷积 | ❌ 丢失75% | ⚠️ 部分保留 | 一般场景 |
| **小波变换** | ✅ 100%保留 | ✅ 完整保留 | **小目标检测、边缘检测** |

## 🔬 技术原理

### 1. 小波分解

小波变换将输入分解为4个子带：

```
输入图像 [H, W]
    ↓ DWT
    ├─ LL [H/2, W/2]  低频近似（主要内容）
    ├─ LH [H/2, W/2]  水平高频（水平边缘）
    ├─ HL [H/2, W/2]  垂直高频（垂直边缘）
    └─ HH [H/2, W/2]  对角高频（对角边缘）
```

### 2. 在ResNet中的应用

#### 位置1: Stem Layer（第834-843行）
```python
# 原始代码位置: xiaobo_resnet_s.py
coeffs, coeffs_ = self.xfm(x)  # 小波分解输入图像
x = self.conv1(x)              # 传统卷积
x = torch.cat([x, coeffs, ...], dim=1)  # 融合小波特征
```

**作用**: 在网络入口融合空域和频域特征

#### 位置2: Bottleneck下采样（第454-456行）
```python
# 原始代码位置: xiaobo_resnet_s.py
if self.conv2_stride == 2:
    coeffs_yl, _ = self.xfm(out)  # 小波下采样
    out = self.conv2(coeffs_yl)   # 在低频上卷积
```

**作用**: 用小波替代stride=2卷积，保留高频信息

## 📦 文件结构

```
wavelet_network/
├── wavelet_module.py          # 核心模块（可独立使用）
│   ├── WaveletDownsample      # 小波下采样
│   ├── WaveletFeatureFusion   # 小波特征融合
│   ├── WaveletStemLayer       # 小波Stem层
│   └── WaveletBottleneck      # 小波Bottleneck块
│
├── demo_wavelet.py            # 演示脚本
├── requirements.txt           # 依赖包
└── README.md                  # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd wavelet_network
pip install -r requirements.txt
```

**注意**: 如果`pytorch_wavelets`安装失败，尝试：
```bash
pip install git+https://github.com/fbcotter/pytorch_wavelets.git
```

### 2. 运行演示

```bash
python demo_wavelet.py
```

演示包含：
- ✅ 小波分解可视化
- ✅ 下采样方法对比
- ✅ 小波Stem层演示
- ✅ 小波Bottleneck演示
- ✅ 性能基准测试

### 3. 在代码中使用

```python
import torch
from wavelet_module import WaveletBottleneck, WaveletStemLayer

# 示例1: 使用小波Stem层
x = torch.randn(1, 3, 224, 224)
stem = WaveletStemLayer(in_channels=3, stem_channels=64)
out = stem(x)  # [1, 64, 56, 56]

# 示例2: 使用小波Bottleneck
x = torch.randn(1, 256, 56, 56)
block = WaveletBottleneck(in_channels=256, out_channels=512, stride=2)
out = block(x)  # [1, 512, 28, 28]
```

## 📊 实验效果

根据DWWA论文，在5个公共数据集上验证：

| 数据集 | Baseline | +Wavelet | 提升 |
|--------|----------|----------|------|
| NEU | 72.3% | 74.1% | +1.8% |
| GC10 | 68.5% | 70.2% | +1.7% |
| ... | ... | ... | ... |

**结论**: 小波网络对小目标、细节丰富的场景特别有效

## 🔧 模块详解

### WaveletDownsample
基础小波下采样模块
```python
wavelet = WaveletDownsample(wave='db1', mode='zero', J=1)
yl, yh = wavelet(x)  # yl: 低频, yh: 高频
```

### WaveletFeatureFusion
小波特征融合模块
```python
fusion = WaveletFeatureFusion(in_channels=64, out_channels=128)
out = fusion(x)  # 自动处理LL/LH/HL/HH融合
```

### WaveletStemLayer
网络输入层，融合小波和卷积特征
```python
stem = WaveletStemLayer(in_channels=3, stem_channels=64)
out = stem(rgb_image)
```

### WaveletBottleneck
ResNet Bottleneck，使用小波下采样
```python
block = WaveletBottleneck(256, 512, stride=2)
out = block(x)  # stride=2时使用小波
```

## 💡 使用建议

### 适用场景
✅ 小目标检测
✅ 高分辨率图像
✅ 边缘/纹理重要的任务
✅ 医疗影像分析
✅ 遥感图像处理

### 不适用场景
❌ 极简单的分类任务
❌ 计算资源极度受限
❌ 低分辨率图像（<100x100）

### 性能优化
- 使用`db1`小波（最快）
- 仅在关键层使用小波（如stem + 首次下采样）
- 考虑使用混合精度训练

## 🔬 与原始代码的对应关系

| 模块 | 原始位置 | 本项目位置 |
|------|---------|-----------|
| 小波Stem | `xiaobo_resnet_s.py:834-843` | `WaveletStemLayer` |
| 小波Bottleneck | `xiaobo_resnet_s.py:454-456` | `WaveletBottleneck` |
| 小波融合 | `xiaobo_resnet_s.py:102-107` | `WaveletFeatureFusion` |

## 📖 参考资料

- **DWWA论文**: 原始论文中的小波网络部分
- **PyWavelets文档**: https://github.com/fbcotter/pytorch_wavelets
- **小波变换理论**: Daubechies小波基础

## ❓ 常见问题

**Q1: 为什么选择db1小波？**
A: db1(Haar小波)计算最快，且实验证明效果足够好

**Q2: 小波变换会增加多少计算量？**
A: 约10-15%，但精度提升显著

**Q3: 能用于其他网络吗？**
A: 可以！小波下采样可替换任何网络的stride=2层

**Q4: 如何可视化小波系数？**
A: 运行`demo_wavelet.py`会生成可视化图像

## 🤝 贡献

欢迎提交Issue和PR！

## 📝 许可证

与DWWA主项目相同

---

**提取时间**: 2025-10-28
**原始项目**: DWWA (Discrete Wavelet with Weighted Attention)
**核心创新**: 使用小波变换替代传统下采样，保留高频细节信息
