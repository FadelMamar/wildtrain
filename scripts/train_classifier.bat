call cd /d %~dp0 && cd ..

call set CONFIG_FILE=configs\classification\example_config.yaml

::call set CONFIG_FILE=pipelines\classification_pipeline_config.yaml

call uv run wildtrain train-classifier %CONFIG_FILE%