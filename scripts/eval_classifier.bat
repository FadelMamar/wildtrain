@echo off
REM Example script to evaluate a classifier using WildTrain CLI

call cd D:\PhD\workspace\wildtrain

REM Set example config path
set CONFIG_PATH=configs\classification\classification_eval.yaml

REM Run the evaluate-classifier CLI command
call uv run wildtrain evaluate-classifier %CONFIG_PATH% 