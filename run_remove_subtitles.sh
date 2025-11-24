#!/bin/bash
# 运行视频去字幕脚本

# 设置项目根目录
cd "$(dirname "$0")"

# 设置环境变量
export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd):$PYTHONPATH

# 运行脚本
python3 bin/remove_subtitles_from_video.py

