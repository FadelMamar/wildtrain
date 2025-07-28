@echo off
REM Example script to run the detection pipeline using WildTrain CLI

call cd /d %~dp0 && cd ..

REM Set example config path
set CONFIG_PATH=pipelines\yolo_pipeline_config.yaml

REM Run the run_classification_pipeline CLI command
call uv run wildtrain run-detection-pipeline ^
    %CONFIG_PATH% 