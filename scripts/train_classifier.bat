call cd D:\workspace\repos\wildtrain
call cd D:\PhD\workspace\wildtrain

call set CONFIG_FILE=configs\classification\example_config.yaml

::call set CONFIG_FILE=pipelines\classification_pipeline_config.yaml

call uv run wildtrain train-classifier %CONFIG_FILE%