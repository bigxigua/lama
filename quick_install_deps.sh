#!/bin/bash
# 快速安装依赖脚本（避免编译和 conda 卡住问题）

set -e

echo "=========================================="
echo "快速安装 LaMa 项目依赖"
echo "=========================================="
echo ""

# 激活环境
if ! conda env list | grep -q "^lama "; then
    echo "错误：lama 环境不存在，请先运行 init_ubuntu_server.sh"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lama

echo "[1/3] 安装 OpenCV（使用 pip 预编译包）..."
# 尝试安装有预编译包的版本
pip install opencv-python==4.5.3.56 --only-binary :all: 2>/dev/null || \
pip install opencv-python-headless==4.5.3.56 --only-binary :all: 2>/dev/null || \
pip install opencv-python 2>/dev/null || {
    echo "⚠️  OpenCV 安装失败，尝试编译安装（可能需要较长时间）..."
    pip install opencv-python-headless
}

echo "[2/3] 安装 Shapely（使用 pip 预编译包）..."
# 尝试安装最新版本（通常有预编译包）
pip install shapely --only-binary :all: 2>/dev/null || \
pip install shapely==1.7.1 --only-binary :all: 2>/dev/null || {
    echo "⚠️  Shapely 需要编译，安装编译依赖..."
    sudo apt-get install -y libgeos-dev 2>/dev/null || true
    pip install shapely==1.7.1
}

echo "[3/3] 安装其他 Python 依赖..."
pip install easydict==1.9.0
pip install albumentations==0.5.2
pip install hydra-core==1.1.0
pip install pytorch-lightning==1.2.9
pip install kornia==0.5.0
pip install webdataset
pip install wldhx.yadisk-direct
pip install braceexpand==0.1.7
pip install imgaug==0.4.0

echo ""
echo "=========================================="
echo "✅ 依赖安装完成！"
echo "=========================================="
echo ""
echo "验证安装："
python -c "import cv2; print(f'  ✅ OpenCV: {cv2.__version__}')" 2>/dev/null || echo "  ❌ OpenCV 未安装"
python -c "import shapely; print('  ✅ Shapely: OK')" 2>/dev/null || echo "  ❌ Shapely 未安装"
python -c "import torch; print(f'  ✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo "  ❌ PyTorch 未安装"
python -c "import pytorch_lightning; print(f'  ✅ Lightning: {pytorch_lightning.__version__}')" 2>/dev/null || echo "  ❌ Lightning 未安装"
echo ""

