call cd /d "%~dp0" && cd ..
call set CONFIG_FILE=configs\datapreparation\savmap.yaml
call uv run --no-sync wildata import-dataset --config %CONFIG_FILE%