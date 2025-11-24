# OpenCV 安装卡住问题解决方案

## 🚨 当前问题

如果 `opencv-python` 安装时卡在 "Building wheel" 步骤，这是因为它在从源码编译，会非常慢（可能需要30分钟到1小时）。

## ⚡ 快速解决方案

### 方案 1：中断并使用 conda 安装（推荐）

1. **中断当前安装**（按 `Ctrl+C`）

2. **使用 conda 安装 OpenCV**（预编译，秒级完成）：
```bash
conda activate lama
conda install -y opencv -c conda-forge
```

3. **继续安装其他依赖**：
```bash
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

### 方案 2：等待编译完成（不推荐）

如果你不想中断，可以等待编译完成，但可能需要30分钟到1小时。

### 方案 3：使用预编译的 wheel 文件

如果 conda 安装失败，可以尝试安装特定版本的预编译包：

```bash
conda activate lama

# 尝试安装较旧但稳定的版本（通常有预编译包）
pip install opencv-python==4.5.3.56 --no-build-isolation

# 或者只安装 headless 版本（无 GUI，但更轻量）
pip install opencv-python-headless==4.5.3.56 --no-build-isolation
```

### 方案 4：跳过 OpenCV，稍后安装

如果以上都失败，可以先跳过 OpenCV，完成其他安装：

```bash
conda activate lama

# 跳过 opencv，继续安装其他包
pip install albumentations==0.5.2
pip install hydra-core==1.1.0
pip install pytorch-lightning==1.2.9
pip install kornia==0.5.0
pip install webdataset
pip install wldhx.yadisk-direct
pip install braceexpand==0.1.7
pip install imgaug==0.4.0
pip install shapely==1.7.1

# 稍后再安装 OpenCV
conda install -y opencv -c conda-forge
```

## ✅ 验证安装

安装完成后验证：

```bash
conda activate lama
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## 📝 为什么会出现这个问题？

- `opencv-python` 在某些系统上需要从源码编译
- 编译需要大量时间和系统资源
- conda 提供的预编译包可以避免这个问题

## 🔧 预防措施

脚本已更新，现在会优先使用 conda 安装 OpenCV，避免编译问题。

