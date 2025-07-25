from wildtrain.data.filters import ClassificationRebalanceFilter
from collections import Counter
import json

# Example annotation data (List[dict])
path = "D:/PhD/workspace/data/framework_formats/roi/wetseason-leopardrock-camp22+37-41-rep3/labels/val/roi_labels.json"
with open(path, "r") as f:
    annotations = json.load(f)


# Print class distribution before filtering
class_ids = [ann["class_id"] for ann in annotations]
print("Class distribution before filtering:", Counter(class_ids))

# Instantiate and apply the filter
filter = ClassificationRebalanceFilter(class_key="class_id", random_seed=42)
balanced_annotations = filter(annotations)

# Print class distribution after filtering
balanced_class_ids = [ann["class_id"] for ann in balanced_annotations]
print("Class distribution after filtering:", Counter(balanced_class_ids))

# Optionally, print the filtered annotations
print("\nFiltered annotations:")
#for ann in balanced_annotations:
#    print(ann)
