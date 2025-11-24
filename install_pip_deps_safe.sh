#!/bin/bash
# 安全安装 pip 依赖（使用清华源，跳过有问题的包）

echo "正在配置 pip 使用清华源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

echo ""
echo "开始安装 pip 依赖包（跳过可能有问题的包）..."
echo ""

# 安装基础包
pip install easydict==1.9.0

# scikit-image 和 scikit-learn 可能已通过 conda 安装，跳过特定版本
# 如果需要特定版本，可以稍后单独处理
echo "跳过 scikit-image 和 scikit-learn（conda 已安装）"

# 安装其他依赖
pip install opencv-python || echo "⚠️ opencv-python 安装失败，可以稍后重试"
pip install albumentations==0.5.2 || echo "⚠️ albumentations 安装失败"
pip install hydra-core==1.1.0 || echo "⚠️ hydra-core 安装失败"
pip install pytorch-lightning==1.2.9 || echo "⚠️ pytorch-lightning 安装失败"
pip install kornia==0.5.0 || echo "⚠️ kornia 安装失败"
pip install webdataset || echo "⚠️ webdataset 安装失败"
pip install wldhx.yadisk-direct || echo "⚠️ wldhx.yadisk-direct 安装失败"
pip install braceexpand==0.1.7 || echo "⚠️ braceexpand 安装失败"
pip install imgaug==0.4.0 || echo "⚠️ imgaug 安装失败"
pip install shapely==1.7.1 || echo "⚠️ shapely 安装失败"
pip install opencv-python-headless==4.5.3.56 || echo "⚠️ opencv-python-headless 安装失败"

# TensorFlow 可选，如果失败可以跳过
pip install tensorflow || echo "⚠️ tensorflow 安装失败（可选，主要用于评估）"

echo ""
echo "✅ pip 依赖安装完成！"
echo ""
echo "如果某些包安装失败，可以稍后单独安装："
echo "  pip install <package_name>"

