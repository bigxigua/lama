# Ubuntu 服务器快速开始指南

## 🚀 一键安装（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/advimman/lama.git
cd lama

# 2. 运行初始化脚本
chmod +x init_ubuntu_server.sh
./init_ubuntu_server.sh

# 3. 重新加载配置
source ~/.bashrc

# 4. 激活环境
conda activate lama
```

## ✅ 验证安装

```bash
conda activate lama
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

## 📥 下载模型

```bash
cd lama
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
```

## 🎯 运行推理

```bash
cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
conda activate lama

python3 bin/predict.py \
    model.path=$(pwd)/big-lama \
    indir=$(pwd)/LaMa_test_images \
    outdir=$(pwd)/output
```

## 📋 安装清单

- [ ] 系统更新完成
- [ ] Miniconda 已安装
- [ ] conda 环境 `lama` 已创建
- [ ] PyTorch 已安装
- [ ] 项目依赖已安装
- [ ] 预训练模型已下载
- [ ] 环境变量已设置

## 🔧 常用命令

```bash
# 激活环境
conda activate lama

# 设置环境变量
export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# 检查 GPU
nvidia-smi

# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## 📚 详细文档

- 完整安装指南：[INSTALL_UBUNTU.md](INSTALL_UBUNTU.md)
- 故障排查：[TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 项目说明：[README.md](README.md)

