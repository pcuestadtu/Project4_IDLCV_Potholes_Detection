import os
import json
import random

DATASET_DIRECTORY = "/dtu/datasets1/02516/potholes"
IMAGES_DIRECTORY = DATASET_DIRECTORY + "/images"

def create_split(output_path,
                 train_ratio=0.7,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 seed=42):

    # Get list of images
    images = [f for f in os.listdir(IMAGES_DIRECTORY)]

    # Reproducible shuffle
    random.seed(seed)
    random.shuffle(images)

    # Compute split indices
    n = len(images)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    # Save JSON file
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Saved split file to {output_path}")
    print({k: len(v) for k, v in splits.items()})


# Create split file
create_split("splits.json")