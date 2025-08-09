call cd /d %~dp0 && cd ..

call set CONFIG_FILE=configs\classification\classification_train.yaml

::call set CONFIG_FILE=pipelines\classification_pipeline_config.yaml

call uv run wildtrain train classifier -c %CONFIG_FILE%