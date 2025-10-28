# 安装指南

## 环境要求

- Python 3.7+
- CUDA 10.2+ (可选，用于GPU加速)

## 安装步骤

### 1. 安装PyTorch

#### CPU版本
```bash
pip install torch torchvision
```

#### GPU版本（推荐）
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

查看更多版本: https://pytorch.org/get-started/locally/

### 2. 安装pytorch_wavelets

```bash
pip install pytorch_wavelets
```

**如果安装失败**，尝试从源码安装：
```bash
pip install git+https://github.com/fbcotter/pytorch_wavelets.git
```

### 3. 安装其他依赖

```bash
cd wavelet_network
pip install -r requirements.txt
```

## 验证安装

```bash
python -c "import torch; import pytorch_wavelets; print('安装成功!')"
```

## 运行演示

```bash
python demo_wavelet.py
```

## 快速测试

```python
import torch
from wavelet_module import WaveletBottleneck

# 创建测试数据
x = torch.randn(1, 256, 56, 56)

# 创建小波Bottleneck
block = WaveletBottleneck(256, 512, stride=2)

# 前向传播
output = block(x)

print(f"输入: {x.shape}")
print(f"输出: {output.shape}")
print("✓ 测试通过!")
```

## 故障排除

### 问题1: pytorch_wavelets安装失败

**解决方案**:
```bash
# 方案1: 从源码安装
git clone https://github.com/fbcotter/pytorch_wavelets.git
cd pytorch_wavelets
pip install -e .

# 方案2: 使用conda
conda install -c conda-forge pywavelets
```

### 问题2: CUDA版本不匹配

**解决方案**:
```bash
# 检查CUDA版本
nvidia-smi

# 卸载当前PyTorch
pip uninstall torch torchvision

# 安装对应CUDA版本的PyTorch
# 参考: https://pytorch.org/get-started/locally/
```

### 问题3: 内存不足

**解决方案**:
- 减小batch size
- 使用CPU版本测试
- 使用混合精度训练

## 在DWWA主项目中使用

如果您想在DWWA主项目中运行完整代码：

```bash
# 回到主目录
cd /home/user/DWWA

# 安装mmdetection依赖
pip install -r requirements/build.txt
pip install -v -e .

# 运行训练
python tools/debug_train.py configs/DWWA-Net/DWWA_Net.py
```

## Docker环境（推荐）

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# 安装依赖
RUN pip install pytorch_wavelets matplotlib numpy

# 复制代码
COPY wavelet_network /workspace/wavelet_network

# 运行演示
CMD ["python", "wavelet_network/demo_wavelet.py"]
```

构建和运行：
```bash
docker build -t wavelet-demo .
docker run --gpus all wavelet-demo
```
