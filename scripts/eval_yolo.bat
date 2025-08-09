@echo off
REM Example script to evaluate a YOLO model using WildTrain CLI

call cd D:\PhD\workspace\wildtrain

REM Set example config path
set CONFIG_PATH=configs\detection\yolo_eval.yaml

REM Run the eval-yolo CLI command
call uv run wildtrain evaluate-detector %CONFIG_PATH% -t yolo