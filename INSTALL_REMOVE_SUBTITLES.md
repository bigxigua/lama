# 视频去字幕脚本依赖安装指南

本文档说明如何安装 `bin/remove_subtitles_from_video.py` 脚本运行所需的所有依赖。

## 快速安装（推荐）

### 方法 1: 使用安装脚本（推荐）

```bash
# 给脚本添加执行权限
chmod +x install_remove_subtitles_deps.sh

# 运行安装脚本
./install_remove_subtitles_deps.sh
```

脚本会自动：
- 检查 Python 版本
- 配置 pip 源（使用清华源）
- 检查 ffmpeg
- 安装所有 Python 依赖
- 验证安装

### 方法 2: 手动安装

#### 1. 安装系统依赖

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

**CentOS/RHEL:**
```bash
sudo yum install -y ffmpeg
```

#### 2. 安装 Python 依赖

```bash
# 升级 pip
pip3 install --upgrade pip

# 基础依赖
pip3 install numpy pyyaml tqdm omegaconf easydict

# OpenCV
pip3 install opencv-python

# PyTorch（根据您的需求选择）
# CPU 版本
pip3 install torch torchvision torchaudio

# 或 CUDA 11.8 版本（需要 NVIDIA GPU）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1 版本（需要 NVIDIA GPU）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch Lightning（saicinpainting 模块需要）
pip3 install pytorch-lightning==1.2.9

# Hydra（saicinpainting 模块可能需要）
pip3 install hydra-core==1.1.0
```

#### 3. 使用国内镜像源（可选，加速下载）

```bash
# 配置 pip 使用清华源
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

## 依赖列表

### 必需依赖

| 包名 | 用途 | 最小版本 |
|------|------|----------|
| `numpy` | 数值计算 | 1.19.0+ |
| `opencv-python` | 图像和视频处理 | 4.5.0+ |
| `torch` | PyTorch 深度学习框架 | 1.8.0+ |
| `pyyaml` | YAML 配置文件解析 | 5.4.0+ |
| `tqdm` | 进度条显示 | 4.60.0+ |
| `omegaconf` | 配置管理 | 2.1.0+ |
| `easydict` | 字典访问工具 | 1.9.0 |
| `pytorch-lightning` | PyTorch Lightning 框架 | 1.2.9 |
| `hydra-core` | Hydra 配置框架 | 1.1.0 |

### 系统工具

- **ffmpeg**: 用于视频和音频处理（提取音频、合并视频等）

### 项目内部模块

脚本还需要项目内部的 `saicinpainting` 模块，这些模块已经包含在项目中：
- `saicinpainting.evaluation.utils`
- `saicinpainting.evaluation.data`
- `saicinpainting.training.trainers`

## 验证安装

运行以下命令验证所有依赖是否正确安装：

```bash
python3 << EOF
import sys
print(f"Python 版本: {sys.version}")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV 导入失败: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy 导入失败: {e}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")

try:
    import yaml
    print(f"✅ PyYAML: {yaml.__version__}")
except ImportError as e:
    print(f"❌ PyYAML 导入失败: {e}")

try:
    import tqdm
    print(f"✅ tqdm: {tqdm.__version__}")
except ImportError as e:
    print(f"❌ tqdm 导入失败: {e}")

try:
    import omegaconf
    print(f"✅ OmegaConf: {omegaconf.__version__}")
except ImportError as e:
    print(f"❌ OmegaConf 导入失败: {e}")

try:
    import easydict
    print(f"✅ EasyDict: {easydict.__version__}")
except ImportError as e:
    print(f"❌ EasyDict 导入失败: {e}")

try:
    import pytorch_lightning
    print(f"✅ PyTorch Lightning: {pytorch_lightning.__version__}")
except ImportError as e:
    print(f"❌ PyTorch Lightning 导入失败: {e}")

try:
    import hydra
    print(f"✅ Hydra: {hydra.__version__}")
except ImportError as e:
    print(f"❌ Hydra 导入失败: {e}")

# 检查项目模块
try:
    sys.path.insert(0, '.')
    from saicinpainting.evaluation.utils import move_to_device
    from saicinpainting.evaluation.data import pad_img_to_modulo
    from saicinpainting.training.trainers import load_checkpoint
    print("✅ saicinpainting 模块导入成功")
except ImportError as e:
    print(f"❌ saicinpainting 模块导入失败: {e}")
    print("   请确保在项目根目录运行此脚本")
EOF
```

## 使用 Conda 环境（可选）

如果您使用 Conda，可以创建独立的环境：

```bash
# 创建环境
conda create -n lama_subtitle python=3.8 -y
conda activate lama_subtitle

# 安装依赖
pip install numpy pyyaml tqdm omegaconf easydict opencv-python
pip install torch torchvision torchaudio  # 或 CUDA 版本
pip install pytorch-lightning==1.2.9 hydra-core==1.1.0

# 安装 ffmpeg
conda install -c conda-forge ffmpeg -y
```

## 常见问题

### 1. PyTorch CUDA 版本不匹配

如果您的系统有 NVIDIA GPU，但 PyTorch 无法使用 CUDA：

1. 检查 CUDA 版本：
   ```bash
   nvidia-smi
   ```

2. 根据 CUDA 版本安装对应的 PyTorch：
   - CUDA 11.8: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
   - CUDA 12.1: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### 2. ffmpeg 未找到

确保已安装 ffmpeg 并在 PATH 中：
```bash
which ffmpeg
ffmpeg -version
```

### 3. saicinpainting 模块导入失败

确保在项目根目录运行脚本：
```bash
cd /path/to/lama
python3 bin/remove_subtitles_from_video.py
```

### 4. 依赖版本冲突

如果遇到版本冲突，建议使用虚拟环境：
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 运行脚本

安装完所有依赖后，运行脚本：

```bash
# 确保在项目根目录
cd /path/to/lama

# 确保有以下文件：
# - pretrained_models/best.ckpt (预训练模型)
# - input_videos/input.mp4 (输入视频)

# 运行脚本
python3 bin/remove_subtitles_from_video.py
```

输出视频将保存在 `output_videos/output.mp4`。

## 更多信息

- 项目主 README: [README.md](README.md)
- 安装问题排查: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

