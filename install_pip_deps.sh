#!/bin/bash
# 手动安装 pip 依赖（使用清华源）

echo "正在配置 pip 使用清华源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

echo ""
echo "开始安装 pip 依赖包..."
echo ""

# 安装依赖包列表
pip install easydict==1.9.0

# scikit-image 0.17.2 可能与新版本 numpy 不兼容，尝试安装兼容版本
# 如果失败，可以跳过版本号让 pip 自动选择兼容版本
pip install "scikit-image>=0.17.2,<0.20" || pip install scikit-image

# scikit-learn 同样处理
pip install "scikit-learn>=0.24.2,<0.25" || pip install scikit-learn
pip install opencv-python
pip install tensorflow
pip install albumentations==0.5.2
pip install hydra-core==1.1.0
pip install pytorch-lightning==1.2.9
pip install kornia==0.5.0
pip install webdataset
pip install wldhx.yadisk-direct
pip install braceexpand==0.1.7
pip install imgaug==0.4.0
pip install shapely==1.7.1
pip install opencv-python-headless==4.5.3.56

echo ""
echo "✅ pip 依赖安装完成！"

