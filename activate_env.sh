#!/bin/bash
# TradingAgents 环境激活脚本
# 使用方法: source activate_env.sh

# 激活 conda 环境
conda activate tradingagents

# 确保使用 conda 环境的 Python 和 pip
export PATH="/opt/homebrew/Caskroom/miniconda/base/envs/tradingagents/bin:$PATH"

# 验证
echo "✓ Conda 环境已激活: tradingagents"
echo "✓ Python 路径: $(which python)"
echo "✓ Pip 路径: $(which pip)"
echo ""
echo "现在可以使用以下命令："
echo "  python lina_test.py"
echo "  pip install <package>"

