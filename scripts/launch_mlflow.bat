call cd /d %~dp0 && cd ..
call uv run mlflow server --backend-store-uri runs\mlflow


