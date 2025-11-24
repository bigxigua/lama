#!/bin/bash
# LaMa 项目完整依赖安装脚本
# 适用于 Ubuntu 24.04 LTS

set -e

echo "=========================================="
echo "LaMa 项目完整依赖安装"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 检查 conda 环境
if ! command -v conda &> /dev/null; then
    echo -e "${RED}错误：conda 未安装或未在 PATH 中${NC}"
    echo "请先运行 init_ubuntu_server.sh 或手动安装 Miniconda"
    exit 1
fi

# 检查环境是否存在
if ! conda env list | grep -q "^lama "; then
    echo -e "${YELLOW}警告：lama 环境不存在，正在创建...${NC}"
    conda create -n lama python=3.8 -y
fi

# 配置 pip 使用清华源（加速下载）
echo -e "${GREEN}[0/6] 配置 pip 和 conda 使用国内镜像源...${NC}"
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 配置 conda 使用清华源（如果还没有配置）
if [ ! -f ~/.condarc ] || ! grep -q "mirrors.tuna.tsinghua.edu.cn" ~/.condarc 2>/dev/null; then
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
    echo "  ✅ conda 已配置使用清华源"
fi

echo "  ✅ pip 已配置使用清华源"

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lama

echo -e "${GREEN}[1/6] 升级 pip...${NC}"
pip install --upgrade pip

echo -e "${GREEN}[2/6] 安装 PyTorch...${NC}"
# 检查是否有 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "检测到 NVIDIA GPU，安装 CUDA 版本..."
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch -y || \
    pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
else
    echo "未检测到 GPU，安装 CPU 版本..."
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch -y || \
    pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
fi

echo -e "${GREEN}[3/6] 安装基础科学计算包（conda，预编译）...${NC}"
conda install -y numpy scipy matplotlib pandas scikit-image scikit-learn joblib pyyaml tqdm tabulate packaging -c conda-forge || {
    echo -e "${YELLOW}conda 安装失败，使用 pip 安装...${NC}"
    pip install numpy scipy matplotlib pandas scikit-image scikit-learn joblib pyyaml tqdm tabulate packaging
}

echo -e "${GREEN}[4/6] 安装 OpenCV 和 Shapely（优先使用预编译包）...${NC}"

# OpenCV
echo "  安装 OpenCV..."
pip install opencv-python-headless==4.5.3.56 --only-binary :all: 2>/dev/null || \
pip install opencv-python==4.5.3.56 --only-binary :all: 2>/dev/null || \
pip install opencv-python-headless --only-binary :all: 2>/dev/null || {
    echo -e "${YELLOW}  使用 conda 安装 OpenCV...${NC}"
    conda install -y opencv -c conda-forge 2>/dev/null || {
        echo -e "${YELLOW}  需要编译 OpenCV（可能需要较长时间）...${NC}"
        pip install opencv-python-headless
    }
}

# Shapely
echo "  安装 Shapely..."
pip install shapely --only-binary :all: 2>/dev/null || \
pip install shapely==1.7.1 --only-binary :all: 2>/dev/null || {
    echo -e "${YELLOW}  安装 Shapely 编译依赖...${NC}"
    sudo apt-get install -y libgeos-dev 2>/dev/null || true
    pip install shapely==1.7.1
}

echo -e "${GREEN}[5/6] 安装项目特定依赖...${NC}"
pip install easydict==1.9.0
pip install albumentations==0.5.2
pip install hydra-core==1.1.0
pip install pytorch-lightning==1.2.9
pip install kornia==0.5.0
pip install webdataset
pip install wldhx.yadisk-direct
pip install braceexpand==0.1.7
pip install imgaug==0.4.0

echo -e "${GREEN}[6/6] 安装可选依赖...${NC}"
# TensorFlow（可选，主要用于评估）
echo "  安装 TensorFlow（可选）..."
pip install tensorflow --only-binary :all: 2>/dev/null || \
echo -e "${YELLOW}  TensorFlow 安装失败（可选包，可稍后安装）${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}✅ 依赖安装完成！${NC}"
echo "=========================================="
echo ""
echo "验证安装："
echo ""

# 验证关键包
python -c "import torch; print(f'  ✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo -e "  ${RED}❌ PyTorch 未安装${NC}"
python -c "import cv2; print(f'  ✅ OpenCV: {cv2.__version__}')" 2>/dev/null || echo -e "  ${RED}❌ OpenCV 未安装${NC}"
python -c "import shapely; print('  ✅ Shapely: OK')" 2>/dev/null || echo -e "  ${RED}❌ Shapely 未安装${NC}"
python -c "import numpy; print(f'  ✅ NumPy: {numpy.__version__}')" 2>/dev/null || echo -e "  ${RED}❌ NumPy 未安装${NC}"
python -c "import pytorch_lightning; print(f'  ✅ PyTorch Lightning: {pytorch_lightning.__version__}')" 2>/dev/null || echo -e "  ${RED}❌ PyTorch Lightning 未安装${NC}"
python -c "import albumentations; print(f'  ✅ Albumentations: {albumentations.__version__}')" 2>/dev/null || echo -e "  ${RED}❌ Albumentations 未安装${NC}"
python -c "import kornia; print(f'  ✅ Kornia: {kornia.__version__}')" 2>/dev/null || echo -e "  ${RED}❌ Kornia 未安装${NC}"

echo ""
echo "下一步："
echo "  1. 设置环境变量："
echo "     export TORCH_HOME=\$(pwd)"
echo "     export PYTHONPATH=\$(pwd)"
echo ""
echo "  2. 下载预训练模型（可选）："
echo "     curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
echo "     unzip big-lama.zip"
echo ""

