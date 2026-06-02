#!/bin/bash

# 启动 Uvicorn 服务
nohup uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 4 > sandbox.log 2>&1 &

# 可选：显示启动提示
echo "Uvicorn 服务已启动，进程 PID: $!"
echo "日志文件: sanbox.log"
echo "查看实时日志: tail -f sandbox.log"
echo "停止服务: pkill -f 'uvicorn sandbox_api:app' 或 kill $!"