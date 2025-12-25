#!/bin/bash

# 获取所有ray::WorkerDict进程的PID
pids=$(ps aux | grep 'ray::WorkerDict' | grep -v grep | awk '{print $2}')

# 检查是否有进程需要终止
if [ -z "$pids" ]; then
    echo "没有找到ray::WorkerDict进程"
    exit 0
fi

# 终止所有找到的进程
echo "正在终止以下ray::WorkerDict进程:"
ps aux | grep 'ray::WorkerDict' | grep -v grep
echo ""

for pid in $pids; do
    echo "正在终止进程 $pid"
    kill $pid
done

echo "所有ray::WorkerDict进程已终止"