# 解决包编译卡住问题

## 🚨 当前问题

如果安装时卡在 "Building wheel" 步骤，说明某些包正在从源码编译，这会非常慢。

常见需要编译的包：
- `opencv-python` / `opencv-python-headless`
- `Shapely`
- `numpy`（如果版本不匹配）
- `scipy`（如果版本不匹配）

## ⚡ 立即解决方案

### 步骤 1：中断当前安装

按 `Ctrl+C` 中断当前安装。

### 步骤 2：使用 conda 安装所有可能编译的包

```bash
# 确保环境已激活
conda activate lama

# 使用 conda 安装所有可能编译的包（预编译，秒级完成）
conda install -y opencv shapely -c conda-forge

# 验证安装
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import shapely; print('Shapely OK')"
```

### 步骤 3：继续安装其他依赖

```bash
# 安装纯 Python 包（不需要编译）
pip install easydict==1.9.0
pip install albumentations==0.5.2
pip install hydra-core==1.1.0
pip install pytorch-lightning==1.2.9
pip install kornia==0.5.0
pip install webdataset
pip install wldhx.yadisk-direct
pip install braceexpand==0.1.7
pip install imgaug==0.4.0

# TensorFlow（可选）
pip install tensorflow || echo "TensorFlow 可选，可跳过"
```

## 🔧 如果 conda 安装失败

### 方案 A：使用预编译的 wheel 文件

```bash
conda activate lama

# 强制使用预编译包，不编译
pip install opencv-python==4.5.3.56 --only-binary :all:
pip install shapely --only-binary :all:
```

### 方案 B：安装系统依赖后编译

```bash
# 安装编译依赖
sudo apt-get install -y \
    build-essential \
    cmake \
    libgeos-dev \
    libproj-dev

# 然后再安装
pip install shapely==1.7.1
pip install opencv-python
```

### 方案 C：使用更新的版本（通常有预编译包）

```bash
conda activate lama

# 使用更新的版本（通常有预编译 wheel）
pip install shapely  # 不指定版本，使用最新稳定版
pip install opencv-python  # 不指定版本
```

## ✅ 完整安装命令（推荐）

```bash
conda activate lama

# 1. 使用 conda 安装所有可能编译的包
conda install -y opencv shapely -c conda-forge

# 2. 安装其他依赖
pip install easydict==1.9.0
pip install albumentations==0.5.2
pip install hydra-core==1.1.0
pip install pytorch-lightning==1.2.9
pip install kornia==0.5.0
pip install webdataset
pip install wldhx.yadisk-direct
pip install braceexpand==0.1.7
pip install imgaug==0.4.0

# 3. 验证
python -c "import cv2, shapely, torch, pytorch_lightning; print('所有关键包已安装')"
```

## 📝 为什么会出现编译问题？

1. **旧版本包**：某些旧版本（如 Shapely 1.7.1）可能没有对应 Python 3.8 的预编译包
2. **系统架构**：某些系统架构可能缺少预编译包
3. **依赖缺失**：缺少编译所需的系统库

## 🎯 预防措施

脚本已更新，现在会：
- ✅ 优先使用 conda 安装所有可能编译的包
- ✅ 使用 `--only-binary` 强制使用预编译包
- ✅ 提供回退方案

## 🔍 检查哪些包需要编译

运行以下命令查看哪些包需要编译：

```bash
pip install --dry-run --report - opencv-python shapely 2>&1 | grep -i "build"
```

如果看到 "Building wheel"，说明需要编译。

