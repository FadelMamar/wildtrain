call cd /d "%~dp0" && cd ..
call set CONFIG_FILE=configs\datapreparation\import-config-example.yaml
call uv run wildata import-dataset --config %CONFIG_FILE%