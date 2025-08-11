from wildtrain.utils.mlflow import load_registered_model
import torch
model,metadata = load_registered_model(alias='demo',name='detector',load_unwrapped=True)

print(model)
print(metadata)


print(model.predict(torch.rand(1,3,640,640)))