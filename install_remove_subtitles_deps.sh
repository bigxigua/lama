#!/bin/bash
# 安装 remove_subtitles_from_video.py 脚本所需的依赖

set -e

echo "=========================================="
echo "安装视频去字幕脚本依赖"
echo "=========================================="
echo ""

# 检查 Python 版本
echo "检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "当前 Python 版本: $python_version"
echo ""

# 配置 pip 使用清华源（可选，如果需要）
USE_TSINGHUA=${USE_TSINGHUA:-1}
if [ "$USE_TSINGHUA" = "1" ]; then
    echo "配置 pip 使用清华源..."
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn 2>/dev/null || true
    echo "✅ pip 源配置完成"
    echo ""
fi

# 检查并安装 ffmpeg
echo "检查 ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version | head -n 1)
    echo "✅ ffmpeg 已安装: $ffmpeg_version"
else
    echo "⚠️  ffmpeg 未安装"
    echo "请根据您的系统安装 ffmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  CentOS/RHEL: sudo yum install ffmpeg"
    echo ""
    read -p "是否继续安装 Python 依赖？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# 核心依赖
echo "=========================================="
echo "安装核心 Python 依赖..."
echo "=========================================="
echo ""

# 基础依赖
echo "1. 安装基础依赖..."
pip3 install --upgrade pip
pip3 install numpy
pip3 install pyyaml
pip3 install tqdm
pip3 install omegaconf
pip3 install easydict
echo "✅ 基础依赖安装完成"
echo ""

# OpenCV
echo "2. 安装 OpenCV..."
pip3 install opencv-python
echo "✅ OpenCV 安装完成"
echo ""

# PyTorch (根据系统选择)
echo "3. 安装 PyTorch..."
echo "请选择 PyTorch 安装方式："
echo "  1) CPU 版本（推荐用于测试）"
echo "  2) CUDA 11.8 版本（需要 NVIDIA GPU 和 CUDA 11.8）"
echo "  3) CUDA 12.1 版本（需要 NVIDIA GPU 和 CUDA 12.1）"
echo "  4) 跳过（如果已安装）"
read -p "请选择 (1-4): " pytorch_choice

case $pytorch_choice in
    1)
        echo "安装 PyTorch CPU 版本..."
        pip3 install torch torchvision torchaudio
        ;;
    2)
        echo "安装 PyTorch CUDA 11.8 版本..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    3)
        echo "安装 PyTorch CUDA 12.1 版本..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    4)
        echo "跳过 PyTorch 安装"
        ;;
    *)
        echo "无效选择，安装 CPU 版本..."
        pip3 install torch torchvision torchaudio
        ;;
esac
echo "✅ PyTorch 安装完成"
echo ""

# PyTorch Lightning (saicinpainting 模块需要)
echo "4. 安装 PyTorch Lightning..."
pip3 install pytorch-lightning==1.2.9
echo "✅ PyTorch Lightning 安装完成"
echo ""

# Hydra (saicinpainting 模块可能需要)
echo "5. 安装 Hydra..."
pip3 install hydra-core==1.1.0
echo "✅ Hydra 安装完成"
echo ""

# 验证安装
echo "=========================================="
echo "验证安装..."
echo "=========================================="
echo ""

python3 << EOF
import sys
print(f"Python 版本: {sys.version}")
print()

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
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        print(f"   GPU 0: {torch.cuda.get_device_name(0)}")
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

print()
print("检查项目模块...")
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

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "使用说明："
echo "1. 确保已安装 ffmpeg（用于视频和音频处理）"
echo "2. 将预训练模型放在: pretrained_models/best.ckpt"
echo "3. 将输入视频放在: input_videos/input.mp4"
echo "4. 运行脚本: python3 bin/remove_subtitles_from_video.py"
echo ""

