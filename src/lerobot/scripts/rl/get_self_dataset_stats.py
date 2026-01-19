import numpy as np
import json
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_PATH = "/home/xuanyuanj/lerobot/datasets_0819/so101_touch_block/"
OUTPUT_FILE = "/home/xuanyuanj/lerobot/datasets_0819/dataset_stats.json"

ds = LeRobotDataset(DATASET_PATH)

# Initialize stats container
stats = {
    "observation.images.front": {"mean": None, "std": None},
    "observation.images.wrist": {"mean": None, "std": None},
    "observation.state": {"min": None, "max": None},
    "action": {"min": None, "max": None},
}

# Image statistics helper
def compute_image_stats(key):
    sum_ = 0
    sum_sq = 0
    count = 0
    for sample in ds:
        img = np.array(sample[key], dtype=np.float32) / 255.0  # Normalize to [0, 1]
        sum_ += img
        sum_sq += img ** 2
        count += 1
    mean = sum_ / count
    std = np.sqrt(sum_sq / count - mean ** 2)
    return mean.tolist(), std.tolist()

# State/action min-max helper
def compute_min_max(key):
    all_data = np.array([sample[key] for sample in ds], dtype=np.float32)
    return all_data.min(axis=0).tolist(), all_data.max(axis=0).tolist()

# Compute statistics
stats["observation.images.front"]["mean"], stats["observation.images.front"]["std"] = compute_image_stats("observation.images.front")
stats["observation.images.wrist"]["mean"], stats["observation.images.wrist"]["std"] = compute_image_stats("observation.images.wrist")
stats["observation.state"]["min"], stats["observation.state"]["max"] = compute_min_max("observation.state")
stats["action"]["min"], stats["action"]["max"] = compute_min_max("action")

# Save to file
with open(OUTPUT_FILE, "w") as f:
    json.dump(stats, f, indent=4)

print(f"Stats saved to {OUTPUT_FILE}")
