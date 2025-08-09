@echo off
REM Example script to visualize classifier predictions in FiftyOne using WildTrain CLI

call cd /d "%~dp0" && cd ..

REM Set example dataset name and checkpoint path
set DATASET_NAME=dryseason-kapiri-camp9-11-rep1-train
set CHECKPOINT_PATH=mlartifacts/547271105631132050/5126e9f91d264de3afbc3c0f709a37fd/artifacts/checkpoint/best_classifier.pt

REM Run the visualize_predictions CLI command
:: call uv run wildtrain visualize classifier-predictions ^
::    %DATASET_NAME% ^
::    --weights %CHECKPOINT_PATH% ^
::    --batch-size 32 ^
::    --device cpu

set CONFIG_PATH=configs/detection/visualization.yaml
call uv run wildtrain visualize detector-predictions ^
    -c %CONFIG_PATH%
