import yaml
import sys
from wildtrain.evaluators.ultralytics import UltralyticsEvaluator


def main():
    config = "D:/workspace/repos/wildtrain/configs/detection/yolo_eval.yaml"
    # Instantiate the evaluator
    evaluator = UltralyticsEvaluator(config=config)

    # Run evaluation
    print("Running YOLO evaluation...")
    results = evaluator.evaluate()

    # Print results
    print("Evaluation results:")
    print(results)


if __name__ == "__main__":
    main()
