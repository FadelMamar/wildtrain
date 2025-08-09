call cd /d "%~dp0" && cd ..

REM Set example config path
set CONFIG_PATH=configs\detection\yolo_eval.yaml

REM Run the evaluate-detector CLI command
call uv run wildtrain evaluate detector -c %CONFIG_PATH% --type yolo