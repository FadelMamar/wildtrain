from wildtrain.utils.mlflow import load_registered_model

model,metadata = load_registered_model(alias='demo',name='detector')

print(model)
print(metadata)