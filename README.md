# WildTrain: Modular Computer Vision Framework

## Overview
WildTrain is a modular, scalable computer vision framework supporting object detection (YOLOv8, MMDetection) and image classification (PyTorch Lightning). It follows Clean Architecture, SOLID principles, and uses Hydra for configuration and MLflow for experiment tracking.

## Project Structure
```
configs/         # Hydra/YAML configs for experiments
models/          # Model definitions (detection/classification)
tasks/           # Task logic (training, evaluation)
trainers/        # Training loops and orchestration
utils/           # Utilities, logging, registry, etc.
data/            # Data modules, transforms, loaders
scripts/         # Example scripts for training/eval
main.py          # CLI entry point
requirements.txt # Python dependencies
pyproject.toml   # Project metadata
```

## Features
- Object Detection: YOLOv8, MMDetection
- Image Classification: PyTorch Lightning
- Hydra-configurable experiments
- MLflow tracking
- Optuna/Ray Tune hyperparameter tuning
- Modular, extensible, type-annotated codebase

## Usage
### Install dependencies
```bash
pip install -r requirements.txt
```

### Train a model
```bash
python main.py task=detection framework=yolov8
python main.py task=classification
```

### Hyperparameter tuning
```bash
python main.py task=classification tune=true
```

### Evaluate
```bash
python main.py task=detection mode=evaluate
```

## Configuration
Edit configs in `configs/` or override via CLI using Hydra syntax.

## Contributing
PRs and issues welcome!