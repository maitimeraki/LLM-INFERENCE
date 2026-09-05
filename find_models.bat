@echo off
REM Helper script to find Hugging Face models and run inference

echo === Hugging Face Model Finder ===
echo.

wsl bash -c "cd /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE && bash scripts/find_models.sh"

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo Copy the model path above and use:
echo   run.bat inference --model ^<path^> --prompt "Hello" --local-files-only
echo.
pause
