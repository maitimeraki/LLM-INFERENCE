@echo off
REM Windows launcher that executes everything in WSL for zero latency
REM
REM Usage from Windows:
REM   run.bat inference --model mistralai/Mixtral-8x7B --prompt "Hello"
REM   run.bat server --model mistralai/Mixtral-8x7B --port 8000

wsl bash -c "cd /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE && bash run_wsl.sh %*"
