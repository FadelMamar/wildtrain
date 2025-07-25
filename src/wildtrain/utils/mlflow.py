import mlflow
import traceback
from pathlib import Path
from typing import Optional, Union
from ..utils.logging import ROOT
from ..utils.logging import get_logger

logger = get_logger(__name__)

# TODO add code to register model


def load_registered_model(
    alias,
    name,
    tag_to_append: str = "",
    mlflow_tracking_url="http://localhost:5000",
    load_unwrapped: bool = False,
    dwnd_location: Optional[Union[str, Path]] = None,
):
    mlflow.set_tracking_uri(mlflow_tracking_url)

    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(name=name, alias=alias).version
    modelversion = f"{name}:{version}" + tag_to_append
    modelURI = f"models:/{name}/{version}"

    if dwnd_location is None:
        dwnd_location = ROOT / Path(f"models/{name}")
        dwnd_location.mkdir(parents=True, exist_ok=True)
        dwnd_location = dwnd_location / version
        dwnd_location = str(dwnd_location.resolve())

    Path(dwnd_location).mkdir(parents=True, exist_ok=True)

    try:
        model = mlflow.pyfunc.load_model(str(dwnd_location))
    except:
        model = mlflow.pyfunc.load_model(modelURI, dst_path=str(dwnd_location))

    metadata = dict(version=modelversion, modeluri=modelURI)
    try:
        metadata.update(model.metadata.metadata)
    except:
        logger.warning(
            f"No metadata found for model {modelversion}. msg {traceback.format_exc()}"
        )

    if load_unwrapped:
        try:
            model = model.unwrap_python_model().model
            metadata["detection_model_type"] = "ultralytics"
        except:
            try:
                model = model.unwrap_python_model().detection_model
                metadata["detection_model_type"] = "ultralytics"
            except:
                model = model.unwrap_python_model().classifier
                metadata["detection_model_type"] = "classifier"

    return model, metadata


def get_experiment_id(name: str):
    """Gets mlflow experiments id

    Args:
        name (str): mlflow experiment name

    Returns:
        str: experiment id
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id
