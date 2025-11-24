# 故障排查指南

## 问题：conda 环境创建时卡在 "Installing pip dependencies"

### 解决方案

#### 方法 1：中断后手动安装（推荐）

**步骤 1：中断当前进程**
```bash
# 按 Ctrl+C 中断当前安装
```

**步骤 2：检查环境是否已创建**
```bash
conda env list
# 如果看到 lama 环境，说明基础环境已创建
```

**步骤 3：激活环境并手动安装 pip 依赖**
```bash
# 激活环境
conda activate lama

# 配置 pip 使用清华源（加速下载）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 使用脚本安装（推荐）
bash install_pip_deps.sh

# 或者手动安装
pip install -r requirements.txt
pip install pytorch-lightning==1.2.9
```

#### 方法 2：使用简化的环境文件

如果方法1失败，可以创建一个不包含pip依赖的版本：

**步骤 1：删除当前环境（如果已创建）**
```bash
conda deactivate
conda env remove -n lama
```

**步骤 2：创建基础环境**
```bash
conda create -n lama python=3.8 -y
conda activate lama
```

**步骤 3：安装 conda 包**
```bash
conda install numpy scipy matplotlib pandas scikit-image scikit-learn joblib pyyaml tqdm tabulate packaging -c conda-forge -y
```

**步骤 4：配置 pip 源并安装依赖**
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
bash install_pip_deps.sh
```

#### 方法 3：分批安装（如果某些包失败）

如果某些包安装失败，可以分批安装：

```bash
conda activate lama

# 配置清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 先安装基础包
pip install easydict==1.9.0
pip install scikit-image==0.17.2
pip install scikit-learn==0.24.2

# 安装 OpenCV（可能较慢）
pip install opencv-python opencv-python-headless==4.5.3.56

# 安装深度学习相关（可选，如果不需要可以跳过）
pip install tensorflow  # 可选，主要用于评估

# 安装其他依赖
pip install albumentations==0.5.2
pip install hydra-core==1.1.0
pip install pytorch-lightning==1.2.9
pip install kornia==0.5.0
pip install webdataset
pip install wldhx.yadisk-direct
pip install braceexpand==0.1.7
pip install imgaug==0.4.0
pip install shapely==1.7.1
```

### 常见问题

#### 1. TensorFlow 安装失败

TensorFlow 主要用于评估指标，如果不需要可以跳过：

```bash
# 跳过 tensorflow，其他包正常安装
pip install easydict==1.9.0 scikit-image==0.17.2 scikit-learn==0.24.2 \
  opencv-python albumentations==0.5.2 hydra-core==1.1.0 \
  pytorch-lightning==1.2.9 kornia==0.5.0 webdataset \
  wldhx.yadisk-direct braceexpand==0.1.7 imgaug==0.4.0 \
  shapely==1.7.1 opencv-python-headless==4.5.3.56
```

#### 2. 网络超时

如果遇到网络超时，可以：

```bash
# 增加超时时间
pip install --default-timeout=100 <package_name>

# 或者使用国内其他镜像源
pip install -i https://mirrors.aliyun.com/pypi/simple/ <package_name>
```

#### 3. 依赖冲突

如果遇到依赖冲突：

```bash
# 查看已安装的包
pip list

# 升级 pip
pip install --upgrade pip

# 使用 --no-deps 跳过依赖检查（谨慎使用）
pip install <package> --no-deps
```

### 验证安装

安装完成后，验证关键包：

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pytorch_lightning; print('Lightning:', pytorch_lightning.__version__)"
```

