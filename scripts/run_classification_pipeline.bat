@echo off
REM Example script to run the classification pipeline using WildTrain CLI

REM Set example config path
set CONFIG_PATH=pipelines\classification_pipeline_config.yaml

REM Run the run_classification_pipeline CLI command
call uv run wildtrain run-classification-pipeline ^
    %CONFIG_PATH% 