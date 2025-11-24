#!/bin/bash
# Ubuntu 24.04 LTS 服务器初始化安装脚本
# 用于 LaMa 图像修复项目

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Ubuntu 24.04 LTS 服务器初始化安装脚本"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否为 root 用户
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}请不要使用 root 用户运行此脚本${NC}"
   echo "请使用普通用户运行，脚本会在需要时请求 sudo 权限"
   exit 1
fi

# 步骤 1: 更新系统
echo -e "${GREEN}[1/8] 更新系统包列表...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# 步骤 2: 安装基础工具
echo -e "${GREEN}[2/8] 安装基础工具...${NC}"
sudo apt-get install -y \
    wget \
    curl \
    git \
    vim \
    nano \
    tmux \
    mc \
    build-essential \
    rsync \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    unzip \
    ca-certificates \
    gnupg \
    lsb-release

# 步骤 3: 检查 GPU 并安装 CUDA（如果存在）
echo -e "${GREEN}[3/8] 检查 GPU 支持...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}检测到 NVIDIA GPU${NC}"
    nvidia-smi
    echo ""
    echo -e "${YELLOW}注意：CUDA 可能需要单独安装${NC}"
    echo "如果尚未安装 CUDA，请访问: https://developer.nvidia.com/cuda-downloads"
    echo ""
else
    echo -e "${YELLOW}未检测到 NVIDIA GPU，将安装 CPU 版本的 PyTorch${NC}"
fi

# 步骤 4: 安装 Miniconda
echo -e "${GREEN}[4/8] 安装 Miniconda...${NC}"
if [ -d "$HOME/miniconda3" ]; then
    echo -e "${YELLOW}Miniconda 已存在，跳过安装${NC}"
else
    # 检查本地是否已有安装包（按优先级检查多个位置）
    MINICONDA_INSTALLER=""
    POSSIBLE_LOCATIONS=(
        "$HOME/Miniconda3-latest-Linux-x86_64.sh"
        "$HOME/Downloads/Miniconda3-latest-Linux-x86_64.sh"
        "$HOME/miniconda_installer.sh"
        "/tmp/Miniconda3-latest-Linux-x86_64.sh"
        "./Miniconda3-latest-Linux-x86_64.sh"
        "$HOME/miniconda3-latest-Linux-x86_64.sh"
    )
    
    # 也检查任何包含 "Miniconda" 和 ".sh" 的文件
    for location in "${POSSIBLE_LOCATIONS[@]}"; do
        if [ -f "$location" ]; then
            MINICONDA_INSTALLER="$location"
            echo -e "${GREEN}找到本地安装包: $location${NC}"
            break
        fi
    done
    
    # 如果没找到，尝试在当前目录和用户目录查找任何 Miniconda*.sh 文件
    if [ -z "$MINICONDA_INSTALLER" ]; then
        FOUND_FILE=$(find "$HOME" -maxdepth 2 -name "Miniconda*.sh" -type f 2>/dev/null | head -1)
        if [ ! -z "$FOUND_FILE" ]; then
            MINICONDA_INSTALLER="$FOUND_FILE"
            echo -e "${GREEN}找到本地安装包: $FOUND_FILE${NC}"
        fi
    fi
    
    # 如果还是没找到，则下载
    if [ -z "$MINICONDA_INSTALLER" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        MINICONDA_INSTALLER="$HOME/miniconda_installer.sh"
        
        echo -e "${YELLOW}未找到本地安装包，开始下载 Miniconda...${NC}"
        echo -e "${YELLOW}提示：如果下载太慢，可以将安装包上传到以下任一位置：${NC}"
        echo "  - $HOME/Miniconda3-latest-Linux-x86_64.sh"
        echo "  - $HOME/Downloads/Miniconda3-latest-Linux-x86_64.sh"
        echo "  - 当前目录: $(pwd)/Miniconda3-latest-Linux-x86_64.sh"
        echo ""
        wget -O "$MINICONDA_INSTALLER" "$MINICONDA_URL"
    fi
    
    echo "使用安装包: $MINICONDA_INSTALLER"
    echo "安装 Miniconda..."
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    
    # 如果是下载的临时文件，删除它；如果是用户提供的本地文件，保留
    if [[ "$MINICONDA_INSTALLER" == "$HOME/miniconda_installer.sh" ]]; then
        rm "$MINICONDA_INSTALLER"
    fi
    
    # 初始化 conda
    "$HOME/miniconda3/bin/conda" init bash
    "$HOME/miniconda3/bin/conda" init zsh
    
    echo -e "${GREEN}Miniconda 安装完成！${NC}"
    echo -e "${YELLOW}请运行以下命令或重新打开终端以激活 conda:${NC}"
    echo "  source ~/.bashrc"
    echo "  或"
    echo "  source ~/.zshrc"
fi

# 添加 conda 到 PATH（当前会话）
export PATH="$HOME/miniconda3/bin:$PATH"

# 步骤 5: 配置 conda 和 pip 使用国内镜像源（加速下载）
echo -e "${GREEN}[5/8] 配置 conda 和 pip 使用国内镜像源...${NC}"

# 配置 conda 使用清华源
mkdir -p ~/.condarc.d
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

# 配置 pip 使用清华源
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

echo -e "${GREEN}镜像源配置完成！${NC}"

# 步骤 6: 创建 conda 环境
echo -e "${GREEN}[6/8] 创建 LaMa conda 环境...${NC}"

# 确保 conda 可用
if ! command -v conda &> /dev/null; then
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

# 检查环境是否已存在
if conda env list | grep -q "^lama "; then
    echo -e "${YELLOW}lama 环境已存在，跳过创建${NC}"
else
    echo "创建 Python 3.8 环境（推荐版本）..."
    conda create -n lama python=3.8 -y
fi

# 激活环境
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lama

# 步骤 7: 安装 PyTorch
echo -e "${GREEN}[7/8] 安装 PyTorch 和相关库...${NC}"

# 检查是否有 CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    if [ ! -z "$CUDA_VERSION" ]; then
        echo -e "${YELLOW}检测到 CUDA 版本: $CUDA_VERSION${NC}"
        echo "安装支持 CUDA 的 PyTorch..."
        # 根据 CUDA 版本安装对应的 PyTorch
        # 注意：PyTorch 1.8.0 需要 CUDA 10.2 或 11.1
        conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch -y || \
        pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
    else
        echo "安装 CPU 版本的 PyTorch..."
        conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch -y || \
        pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    fi
else
    echo "安装 CPU 版本的 PyTorch..."
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch -y || \
    pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
fi

# 步骤 8: 安装项目依赖
echo -e "${GREEN}[8/8] 安装项目依赖包...${NC}"

# 升级 pip
pip install --upgrade pip

# 安装基础科学计算包（通过 conda，预编译，避免编译）
echo "安装基础科学计算包（conda）..."
conda install -y numpy scipy matplotlib pandas scikit-image scikit-learn joblib pyyaml tqdm tabulate packaging -c conda-forge

# 安装可能编译的包（优先使用 conda）
echo "安装 OpenCV（conda，预编译）..."
conda install -y opencv -c conda-forge || {
    echo -e "${YELLOW}conda 安装 OpenCV 失败，尝试 pip 安装预编译包...${NC}"
    # 尝试安装有预编译 wheel 的版本
    pip install opencv-python==4.5.3.56 --only-binary :all: || \
    pip install opencv-python-headless==4.5.3.56 --only-binary :all: || {
        echo -e "${YELLOW}警告：OpenCV 可能需要编译，这可能需要较长时间...${NC}"
        pip install opencv-python || pip install opencv-python-headless
    }
}

# 安装 Shapely（优先使用 conda，避免编译）
echo "安装 Shapely（conda，预编译）..."
conda install -y shapely -c conda-forge || {
    echo -e "${YELLOW}conda 安装 Shapely 失败，尝试 pip 安装预编译包...${NC}"
    # 尝试安装较新版本（通常有预编译包）
    pip install shapely --only-binary :all: || \
    pip install shapely==1.7.1 --only-binary :all: || {
        echo -e "${YELLOW}警告：Shapely 可能需要编译...${NC}"
        pip install shapely==1.7.1
    }
}

# 安装纯 Python 包（通常不需要编译）
echo "安装纯 Python 依赖包..."
pip install easydict==1.9.0
pip install albumentations==0.5.2
pip install hydra-core==1.1.0
pip install pytorch-lightning==1.2.9
pip install kornia==0.5.0
pip install webdataset
pip install wldhx.yadisk-direct
pip install braceexpand==0.1.7
pip install imgaug==0.4.0

# TensorFlow（可选，主要用于评估）
echo "安装 TensorFlow（可选）..."
pip install tensorflow --only-binary :all: || echo -e "${YELLOW}TensorFlow 安装失败（可选包，可稍后安装）${NC}"

# 完成
echo ""
echo -e "${GREEN}=========================================="
echo "✅ 安装完成！"
echo "==========================================${NC}"
echo ""
echo "下一步操作："
echo ""
echo "1. 重新加载 shell 配置："
echo "   source ~/.bashrc   # 或 source ~/.zshrc"
echo ""
echo "2. 激活 conda 环境："
echo "   conda activate lama"
echo ""
echo "3. 设置项目环境变量："
echo "   cd /path/to/lama"
echo "   export TORCH_HOME=\$(pwd)"
echo "   export PYTHONPATH=\$(pwd)"
echo ""
echo "4. 验证安装："
echo "   python -c \"import torch; print(f'PyTorch: {torch.__version__}')\""
echo "   python -c \"import cv2; print(f'OpenCV: {cv2.__version__}')\""
echo "   python -c \"import pytorch_lightning; print(f'Lightning: {pytorch_lightning.__version__}')\""
echo ""
echo "5. 下载预训练模型（可选）："
echo "   curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
echo "   unzip big-lama.zip"
echo ""
echo -e "${YELLOW}注意：如果服务器有 GPU，请确保已正确安装 CUDA 驱动${NC}"
echo ""

