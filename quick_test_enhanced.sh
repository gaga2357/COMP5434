#!/bin/bash
cd /Users/gaga/CodeBuddy/project/COMP5434

echo "开始增强版训练..."
echo "预计需要5-10分钟，请耐心等待..."

nohup python3 src/train_enhanced.py > training_enhanced.log 2>&1 &
PID=$!

echo "训练进程已启动，PID: $PID"
echo "日志文件: training_enhanced.log"
echo ""
echo "查看实时日志: tail -f training_enhanced.log"
echo "停止训练: kill $PID"
