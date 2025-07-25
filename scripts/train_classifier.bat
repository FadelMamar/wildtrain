call cd D:\workspace\repos\wildtrain
call cd D:\PhD\workspace\wildtrain

call set CONFIG_FILE=pipelines\classification_pipeline_config.yaml

call uv run wildtrain train-classifier --config %CONFIG_FILE%