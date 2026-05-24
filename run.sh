#!/bin/bash
# COSMIC 功能点分解器启动脚本
# 用法: bash run.sh "你的需求"
# 或者: bash run.sh (交互模式)

VENV_PY="/c/Users/mingm_j8zetfq/Documents/cosmic-langgraph/.venv/Scripts/python.exe"
SCRIPT="/c/Users/mingm_j8zetfq/Documents/cosmic-langgraph/cosmic_workflow.py"

cd "$(dirname "$SCRIPT")"
PYTHONHOME="" "$VENV_PY" "$SCRIPT" "$@"
