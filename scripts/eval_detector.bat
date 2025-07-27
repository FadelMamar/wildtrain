call cd /d "%~dp0" && cd ..

REM Set example config path
set CONFIG_PATH=configs\detection\yolo_eval.yaml

REM Run the evaluate-classifier CLI command
call uv run wildtrain evaluate-detector %CONFIG_PATH% --type yolo