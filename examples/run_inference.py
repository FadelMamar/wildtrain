from wildtrain.models.detector import Detector
import torch

url = "http://localhost:4141/predict"
data = torch.rand(1,3,224,224)
out = Detector.predict_inference_service(data,url=url)
print(out)








