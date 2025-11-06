#!/bin/bash

echo "========================================"
echo "启动终极优化训练 - 目标88%准确率"
echo "========================================"
echo ""

cd "$(dirname "$0")"

nohup python3 src/train_ultra.py > training_ultra.log 2>&1 &

PID=$!
echo "训练已在后台启动 (PID: $PID)"
echo "查看实时日志: tail -f training_ultra.log"
echo "查看最新50行: tail -50 training_ultra.log"
echo ""
echo "预计训练时间: 10-15分钟"
echo "========================================"
