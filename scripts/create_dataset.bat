call cd D:\workspace\repos\wildtrain
call set CONFIG_FILE=configs\datapreparation\savmap.yaml
call uv run --no-sync wildata import-dataset --config %CONFIG_FILE%