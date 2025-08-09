call cd /d "%~dp0" && cd ..

@REM call uv run --no-sync wildtrain register-detector --config configs/registration/detector_registration_example.yaml

call uv run --no-sync wildtrain register-classifier --config configs/registration/classifier_registration_example.yaml
