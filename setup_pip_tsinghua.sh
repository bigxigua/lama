#!/bin/bash
# 配置 pip 使用清华源

echo "正在配置 pip 使用清华源..."

# 创建 pip 配置目录
mkdir -p ~/.pip

# 配置 pip 使用清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

echo "✅ pip 已配置为使用清华源"
echo ""
echo "验证配置："
pip config list

