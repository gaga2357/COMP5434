#!/bin/bash

# COMP5434 Dataset3 快速测试脚本

echo "=================================="
echo "COMP5434 Dataset3 Quick Test"
echo "=================================="
echo ""

# 检查Python环境
echo "Checking Python environment..."
python3 --version
echo ""

# 检查并安装依赖
echo "Installing dependencies..."
pip3 install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# 运行测试
echo "Running algorithm verification..."
echo ""
python3 test_solution.py

echo ""
echo "=================================="
echo "Quick test completed!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Prepare your train.csv and test.csv files"
echo "2. Run: python3 dataset3_solution.py (Neural Network)"
echo "   or:  python3 dataset3_logistic.py (Logistic Regression)"
echo "3. Submit the generated submission.csv to Kaggle"
echo ""
