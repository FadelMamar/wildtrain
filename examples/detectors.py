# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:28:12 2025

@author: FADELCO
"""

# import os

import torch
import wildtrain
from wildtrain.models.localizer import UltralyticsLocalizer
from wildtrain.models.classifier import GenericClassifier
from wildtrain.models.detector import Detector
from wildtrain.data import load_image
from PIL import Image

import supervision as sv

# Example label map for classifier
# label_to_class_map = {0: "cat", 1: "dog"}

# Instantiate the localizer (YOLO weights path or model name required)
device = "cpu"
localizer = UltralyticsLocalizer(weights="D:/workspace/repos/wildtrain/models/best.pt", 
                                 conf_thres=0.2,
                                 iou_thres=0.5,
                                 imgsz=800,
                                 device=device)

# Instantiate the classifier
classifier = GenericClassifier.load_from_checkpoint("D:/workspace/repos/wildtrain/models/best_classifier.pt",
                                                    map_location=device
                                                    )

# Create the two-stage detector
classifier = None
model = Detector(localizer=localizer, classifier=classifier)

# Dummy input: batch of 2 RGB images, 3x224x224
path = r"D:\workspace\data\demo-dataset\savmap\images\train\00a033fefe644429a1e0fcffe88f8b39_0_augmented_0_tile_12_832_832.jpg"
images = load_image(path).unsqueeze(0) / 255.

# Run detection
detections = model.predict(images)

# Print results
for i, det in enumerate(detections):
    print(f"Image {i} detections:")
    print(det)
    
box_annotator = sv.BoxAnnotator()

annotated_frame = box_annotator.annotate(
    scene=Image.open(path).convert("RGB"),
    detections=detections[0]
)
