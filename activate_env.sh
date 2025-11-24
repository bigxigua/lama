#!/bin/bash
# 激活 lama conda 环境的辅助脚本

# 加载 conda
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh

# 激活环境
conda activate lama

# 显示环境信息
echo "✅ lama 环境已激活"
echo "Python 版本: $(python --version)"
echo ""
echo "现在可以运行安装命令："
echo "  bash install_pip_deps.sh"
echo "或者"
echo "  bash install_pip_deps_safe.sh"

