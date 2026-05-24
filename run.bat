@echo off
REM COSMIC 功能点分解器 - Windows 批处理启动脚本
setlocal

set VENV_PY=C:\Users\mingm_j8zetfq\Documents\cosmic-langgraph\.venv\Scripts\python.exe
set SCRIPT=C:\Users\mingm_j8zetfq\Documents\cosmic-langgraph\cosmic_workflow.py

cd /d C:\Users\mingm_j8zetfq\Documents\cosmic-langgraph
set PYTHONHOME=
"%VENV_PY%" "%SCRIPT%" %*

endlocal
