call cd /d %~dp0 && cd ..
call uv run --no-sync mlflow server --backend-store-uri runs\mlflow --host 0.0.0.0 --port 5000


