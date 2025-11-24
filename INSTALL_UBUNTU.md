# Ubuntu 24.04 LTS 服务器初始化指南

本指南将帮助你在新购买的 Ubuntu 24.04 LTS 服务器上初始化并安装 LaMa 图像修复项目所需的所有依赖。

## 📋 前置要求

- Ubuntu 24.04 LTS 服务器
- 具有 sudo 权限的用户账户
- 至少 10GB 可用磁盘空间（推荐 20GB+）
- 稳定的网络连接

## 🚀 快速开始

### 方法 1：使用自动化脚本（推荐）

1. **克隆项目到服务器**：
```bash
git clone https://github.com/advimman/lama.git
cd lama
```

2. **运行初始化脚本**：
```bash
chmod +x init_ubuntu_server.sh
./init_ubuntu_server.sh
```

脚本会自动完成以下操作：
- ✅ 更新系统包
- ✅ 安装基础工具（git, wget, ffmpeg 等）
- ✅ 检测并配置 GPU（如果存在）
- ✅ 安装 Miniconda
- ✅ 配置国内镜像源（加速下载）
- ✅ 创建 Python 3.8 环境
- ✅ 安装 PyTorch（CPU/GPU）
- ✅ 安装所有项目依赖

3. **重新加载 shell 配置**：
```bash
source ~/.bashrc
# 或
source ~/.zshrc
```

4. **激活环境并验证**：
```bash
conda activate lama
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### 方法 2：手动安装

如果你更喜欢手动控制每个步骤，可以按照以下步骤操作：

#### 步骤 1：更新系统

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

#### 步骤 2：安装基础工具

```bash
sudo apt-get install -y \
    wget curl git vim nano tmux mc \
    build-essential rsync \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1 \
    ffmpeg unzip ca-certificates
```

#### 步骤 3：安装 Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh

# 初始化 conda
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

#### 步骤 4：配置镜像源（加速下载）

**配置 conda 使用清华源**：
```bash
cat > ~/.condarc << 'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
```

**配置 pip 使用清华源**：
```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

#### 步骤 5：创建 conda 环境

```bash
conda create -n lama python=3.8 -y
conda activate lama
```

#### 步骤 6：安装 PyTorch

**如果有 NVIDIA GPU**：
```bash
# 检查 CUDA 版本
nvidia-smi

# 安装支持 CUDA 的 PyTorch（根据你的 CUDA 版本选择）
# CUDA 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch -y

# 或 CUDA 11.1
pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

**如果只有 CPU**：
```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch -y
```

#### 步骤 7：安装项目依赖

```bash
cd /path/to/lama

# 升级 pip
pip install --upgrade pip

# 安装基础科学计算包
conda install -y numpy scipy matplotlib pandas scikit-image scikit-learn joblib pyyaml tqdm tabulate packaging -c conda-forge

# 安装其他依赖
pip install -r requirements.txt
pip install pytorch-lightning==1.2.9
```

## 🎯 GPU 支持（可选）

如果你的服务器有 NVIDIA GPU，需要安装 CUDA：

### 检查 GPU

```bash
lspci | grep -i nvidia
```

### 安装 NVIDIA 驱动和 CUDA

1. **添加 NVIDIA 仓库**：
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
```

2. **安装 NVIDIA 驱动**：
```bash
sudo apt-get install -y nvidia-driver-535  # 或更新版本
```

3. **安装 CUDA Toolkit**：
```bash
# 访问 https://developer.nvidia.com/cuda-downloads 获取最新安装命令
# 或使用 conda 安装（推荐，更简单）
conda install cudatoolkit=10.2 -c conda-forge
```

4. **验证安装**：
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 📦 设置项目环境

安装完成后，设置项目环境变量：

```bash
cd /path/to/lama
export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)
```

可以将这些环境变量添加到 `~/.bashrc` 或 `~/.zshrc`：

```bash
echo 'export TORCH_HOME=$(pwd)' >> ~/.bashrc
echo 'export PYTHONPATH=$(pwd)' >> ~/.bashrc
```

## ✅ 验证安装

运行以下命令验证所有组件是否正确安装：

```bash
conda activate lama
cd /path/to/lama

# 检查 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查 OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# 检查 PyTorch Lightning
python -c "import pytorch_lightning; print(f'Lightning: {pytorch_lightning.__version__}')"

# 检查其他关键包
python -c "import numpy, scipy, matplotlib, pandas, sklearn, skimage; print('所有基础包已安装')"
```

## 📥 下载预训练模型

下载预训练模型用于推理：

```bash
cd /path/to/lama

# 下载最佳模型（Big LaMa）
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip

# 或从 Google Drive 下载所有模型
# https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips?usp=sharing
```

## 🐳 使用 Docker（可选）

如果你更喜欢使用 Docker，项目提供了 Dockerfile：

```bash
cd docker
bash build.sh  # 构建 Docker 镜像
bash 2_predict_with_gpu.sh /path/to/model /path/to/input /path/to/output
```

## 🔧 常见问题

### 1. conda 命令未找到

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc
```

### 2. PyTorch CUDA 不可用

- 确保已安装 NVIDIA 驱动：`nvidia-smi`
- 确保 CUDA 版本匹配：PyTorch 1.8.0 需要 CUDA 10.2 或 11.1
- 重新安装匹配的 PyTorch 版本

### 3. 某些包安装失败

- 检查网络连接
- 尝试使用不同的镜像源
- 单独安装失败的包：`pip install <package_name>`

### 4. 内存不足

- 使用 CPU 版本的 PyTorch
- 减少 batch size
- 使用更小的模型

### 5. FFmpeg 未找到

```bash
sudo apt-get install -y ffmpeg
```

## 📚 下一步

安装完成后，你可以：

1. **运行推理**：查看 [README.md](README.md) 中的推理部分
2. **训练模型**：查看 [README.md](README.md) 中的训练部分
3. **处理视频**：使用 `bin/remove_subtitles_from_video.py` 处理视频

## 📞 获取帮助

如果遇到问题：

1. 查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. 查看项目 [Issues](https://github.com/advimman/lama/issues)
3. 阅读项目 [README.md](README.md)

## 📝 系统要求总结

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| CPU | 2 核 | 4+ 核 |
| 内存 | 4GB | 8GB+ |
| 磁盘 | 10GB | 20GB+ |
| GPU | 可选 | NVIDIA GPU (4GB+ VRAM) |
| CUDA | - | 10.2 或 11.1 |
| Python | 3.6+ | 3.8 |
| 操作系统 | Ubuntu 18.04+ | Ubuntu 24.04 LTS |

---

**祝使用愉快！** 🎉

