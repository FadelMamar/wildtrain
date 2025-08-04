@echo off
echo 🚀 Starting WildTrain Streamlit UI...
echo.


REM Run the UI
call uv run streamlit run src/ui.py --server.port 8541 --server.address localhost

if errorlevel 1 (
    echo ❌ Error running Streamlit UI
    pause
    exit /b 1
)

echo 👋 Streamlit UI stopped
pause 