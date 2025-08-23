call cd /d "%~dp0" && cd ..

call set MLFLOW_TRACKING_URI=http://127.0.0.1:5000

call uv run wildtrain run-server --config configs/inference.yaml

:: call uv run mlflow models serve -m models:/detector/9 -p 4141 --no-conda --workers 1 --host 0.0.0.0