@echo off
REM SparseLLM Production Server Launcher
REM Starts FastAPI server with vLLM integration for OpenAI-compatible endpoints
REM Usage: start_server.bat [model] [port]

echo ====================================================================
echo SparseLLM Production Server
echo ====================================================================
echo.

REM Check if .env exists
if not exist .env (
    echo [ERROR] .env file not found!
    echo Please create .env with your HF_TOKEN:
    echo.
    echo HF_TOKEN=your_huggingface_token_here
    echo.
    pause
    exit /b 1
)

REM Check if vLLM is installed
python -c "import vllm" 2>nul
if errorlevel 1 (
    echo [ERROR] vLLM not installed!
    echo Install with: pip install vllm
    echo.
    pause
    exit /b 1
)

REM Check if FastAPI is installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo [ERROR] FastAPI not installed!
    echo Install with: pip install fastapi uvicorn
    echo.
    pause
    exit /b 1
)

REM Set default configuration
set MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
set HOST=0.0.0.0
set PORT=8000
set TENSOR_PARALLEL=1
set GPU_MEMORY_UTIL=0.9

REM Override with command line arguments
if not "%1"=="" set MODEL=%1
if not "%2"=="" set PORT=%2

echo Configuration:
echo   Model: %MODEL%
echo   Host: %HOST%:%PORT%
echo   Tensor Parallel Size: %TENSOR_PARALLEL%
echo   GPU Memory Utilization: %GPU_MEMORY_UTIL%
echo.
echo Starting server...
echo ====================================================================
echo.
echo Server will be available at:
echo   - http://localhost:%PORT%/v1/chat/completions
echo   - http://localhost:%PORT%/v1/completions
echo   - http://localhost:%PORT%/health
echo   - http://localhost:%PORT%/metrics
echo.
echo Press Ctrl+C to stop the server
echo ====================================================================
echo.

REM Run serve.py
python serve.py ^
    --model "%MODEL%" ^
    --host %HOST% ^
    --port %PORT% ^
    --tensor-parallel-size %TENSOR_PARALLEL% ^
    --gpu-memory-utilization %GPU_MEMORY_UTIL%

pause
