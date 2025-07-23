import fiftyone as fo
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from wildtrain.data.visualization import FiftyOneManager

# Example configuration
DATASET_NAME = "wildtrain_classification_example"
DATA_ROOT = "D:/workspace/data/demo-dataset"  # Update this to your data root
CHECKPOINT_PATH = "D:/workspace/repos/wildtrain/checkpoints/classification/best-v5.ckpt"  # Update this to your checkpoint
CONFIG_PATH = None  # Or path to your OmegaConf YAML config
SPLIT = "val"  # or "train"/"test"


if __name__ == "__main__":
    # 1. Create FiftyOneManager
    manager = FiftyOneManager(dataset_name=DATASET_NAME,persistent=False)

    # 2. Import classification dataset (from config or direct args)
    manager.import_classification_datamodule(
        root_data_directory=DATA_ROOT,
        config_path=CONFIG_PATH,
        split=SPLIT,
        load_as_single_class=True,
        background_class_name="background",
        single_class_name="wildlife",
        keep_classes=None,
        discard_classes=None,
    )

    # 3. Add predictions from classifier checkpoint
    manager.add_predictions_from_classifier(
        checkpoint_path=CHECKPOINT_PATH,
        prediction_field="predictions",
        batch_size=32,
        device="cpu",
        debug=True
    )

    # 4. Launch FiftyOne app for visualization
    session = fo.launch_app(manager.dataset)
    print(f"FiftyOne app launched for dataset: {manager.dataset_name}")
    session.wait() 