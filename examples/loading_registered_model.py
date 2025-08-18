from wildtrain.utils.mlflow import load_registered_model
from wildtrain.models.detector import Detector
import torch
from omegaconf import OmegaConf


def load_detector():
    model,metadata = load_registered_model(alias='demo',name='detector',load_unwrapped=True)

    print(model.classifier)
    print(metadata)

    try:
        print(model.predict(torch.rand(1,3,640,640)))
    except Exception as e:
        print(model(torch.rand(1,3,640,640)))

def load_from_config():
    cfg = OmegaConf.load("models-registry/detector/9/artifacts/detector_registration_example.yaml")
    localizer_cfg = cfg.localizer.yolo
    localizer_cfg.weights = "models-registry/detector/9/artifacts/best.pt"
    model = Detector.from_config(localizer_config=localizer_cfg,
                                        classifier_ckpt=r"models-registry\detector\9\artifacts\best.ckpt")
    print(model.localizer is None,model.classifier is None)


# load_from_config()

load_detector()