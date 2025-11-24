# macOS 环境安装指南

由于原始的 `conda_env.yml` 文件包含 Linux 特定的构建版本，在 macOS 上无法使用。以下是 macOS 的安装方法：

## 配置 pip 使用清华源（推荐，加速下载）

在开始安装之前，建议先配置 pip 使用清华源以加速下载：

**方法 A：使用 pip config 命令（推荐）**
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

**方法 B：复制配置文件**
```bash
# 创建 pip 配置目录
mkdir -p ~/.pip

# 复制项目中的 pip.conf 文件
cp pip.conf ~/.pip/pip.conf
```

**方法 C：临时使用（单次安装）**
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package_name>
```

## 方法 1：使用简化的 Conda 环境（推荐）

### 步骤 1：创建 Conda 环境

```bash
cd lama
conda env create -f conda_env_macos.yml
conda activate lama

# 配置 pip 使用清华源（如果还没配置）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

### 步骤 2：安装 PyTorch

**CPU 版本：**
```bash
conda install pytorch torchvision torchaudio -c pytorch -y
```

**macOS GPU 版本（M1/M2/M3）：**
```bash
conda install pytorch torchvision torchaudio -c pytorch -y
# 或者使用 pip（如果 conda 版本不可用）
pip install torch torchvision torchaudio
```

**Intel Mac：**
```bash
conda install pytorch torchvision torchaudio -c pytorch -y
```

### 步骤 3：安装 PyTorch Lightning

```bash
pip install pytorch-lightning==1.2.9
```

### 步骤 4：设置环境变量

```bash
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```

## 方法 2：手动创建环境（如果方法1失败）

### 步骤 1：创建基础环境

```bash
conda create -n lama python=3.8 -y
conda activate lama
```

### 步骤 2：安装基础科学计算包

```bash
conda install numpy scipy matplotlib pandas scikit-image scikit-learn -c conda-forge -y
```

### 步骤 3：安装 PyTorch

```bash
# CPU 版本
conda install pytorch torchvision torchaudio -c pytorch -y

# 或者使用 pip
pip install torch torchvision torchaudio
```

### 步骤 4：配置 pip 使用清华源（如果还没配置）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

### 步骤 5：安装其他依赖

```bash
pip install -r requirements.txt
pip install pytorch-lightning==1.2.9
```

### 步骤 5：设置环境变量

```bash
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```

## 方法 3：使用虚拟环境（最简单）

如果你不想使用 conda，可以使用 Python 虚拟环境：

```bash
# 创建虚拟环境
python3 -m venv lama_env
source lama_env/bin/activate

# 配置 pip 使用清华源（推荐）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装 PyTorch
pip install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
pip install pytorch-lightning==1.2.9

# 设置环境变量
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```

## 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import pytorch_lightning; print(f'PyTorch Lightning version: {pytorch_lightning.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## 常见问题

### 1. PyTorch Lightning 版本问题

如果遇到版本兼容性问题，可以尝试：
```bash
pip install pytorch-lightning==1.2.9 --force-reinstall
```

### 2. OpenCV 安装问题

如果 opencv-python 安装失败，尝试：
```bash
pip install opencv-python-headless
```

### 3. TensorFlow 安装问题（可选）

TensorFlow 主要用于评估指标。如果不需要，可以从 requirements.txt 中移除。

对于 Apple Silicon (M1/M2/M3) Mac：
```bash
pip install tensorflow-macos tensorflow-metal
```

对于 Intel Mac：
```bash
pip install tensorflow
```

## 注意事项

1. **Python 版本**：推荐使用 Python 3.8，因为 Python 3.6 已经过时且很多包不再支持
2. **CUDA**：macOS 不支持 NVIDIA CUDA，所以不需要安装 cudatoolkit
3. **MPS（Metal Performance Shaders）**：Apple Silicon Mac 可以使用 MPS 加速，PyTorch 1.12+ 支持

