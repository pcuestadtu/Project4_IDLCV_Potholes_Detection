import os
import json
import numpy as np
import matplotlib.pyplot as plt
from visualize import read_content



def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def read_json(path):
    with open(path) as f:
        json_file = json.load(f)
    return json_file



SELECTIVE_SEARCH_EXTRACTOR="SS"
EDGEBOXES="EdgeBoxes"
EXTRACTOR=SELECTIVE_SEARCH_EXTRACTOR

DATASET_DIR = "/dtu/datasets1/02516/potholes"
IMAGES_DIR = DATASET_DIR + "/images"
ANNOT_DIR=DATASET_DIR+"/annotations"
PROPOSALS_DIR=f"object_proposals_{EXTRACTOR}"

IOU_LABEL_THRESHOLD=0.5

splits = read_json("splits.json")
all_recalls = []

folder_name = f"labeled_object_proposals_{EXTRACTOR}"
os.makedirs(folder_name, exist_ok=True)

for i, img_filename in enumerate(splits["train"]):
    print(f"{EXTRACTOR}: {i}/{len(splits['train'])}")
    img_path = os.path.join(IMAGES_DIR, img_filename)

    # --- 1) Extract proposals ---
    # Get the proposals for this image
    proposal_path = os.path.join(PROPOSALS_DIR, img_filename.replace(".png", ".json"))
    proposals = read_json(proposal_path)

    # --- 2) Load ground truth ---
    annotation_path = os.path.join(ANNOT_DIR, img_filename.replace(".png", ".xml"))
    _, gt_boxes = read_content(annotation_path)

    iou_matrix = np.zeros((len(gt_boxes), len(proposals)))

    # --- 3) Label proposals ---
    labeled_proposals = []
    for j, p in enumerate(proposals):
        ious = [compute_iou(p, gt) for gt in gt_boxes]

        #   Fill IoU matrix (one column per proposal)
        for g_idx, iou in enumerate(ious):
            iou_matrix[g_idx, j] = iou

        max_iou = max(ious) if ious else 0.0
        matched_gt = int(np.argmax(ious)) if ious else -1

        label = "pothole" if max_iou >= IOU_LABEL_THRESHOLD else "background"

        matches = [gt_idx for gt_idx, iou in enumerate(ious) if iou >= IOU_LABEL_THRESHOLD]

        labeled_proposals.append({
            "box": p,
            "label": label
        })

    # --- 4) Compute recall for this image

    recalls = []
    for k in range(1, len(proposals) + 1):
        props = iou_matrix[:, :k]            # first k proposals
        hits = (props >= IOU_LABEL_THRESHOLD).any(axis=1).sum()
        recall_k = hits / len(gt_boxes)
        recalls.append(recall_k)

    all_recalls.append(recalls)            

    # --- 5) Save JSON ---
    output_path=os.path.join(folder_name, img_filename.replace(".png", ".json"))
    with open(output_path, "w") as f:
        json.dump(labeled_proposals, f, indent=4)

print(f"Saved lableled {EXTRACTOR} proposals to folder: {folder_name}")

# --------------------------
# Compute mean recall curve
# --------------------------

# Some images may have fewer proposals than others; pad with last recall value
max_len = max(len(r) for r in all_recalls)
all_recalls_padded = []

for r in all_recalls:
    if len(r) < max_len:
        r = np.pad(r, (0, max_len - len(r)), mode='edge')
    all_recalls_padded.append(r)

all_recalls_padded = np.array(all_recalls_padded)
mean_recall = np.mean(all_recalls_padded, axis=0)

# --------------------------
# Plot mean recall curve
# --------------------------
plot_folder_name = f"plots"
os.makedirs(plot_folder_name, exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(mean_recall)+1), mean_recall)
plt.xlabel("Number of proposals")
plt.ylabel(f"Mean Recall with {SELECTIVE_SEARCH_EXTRACTOR} (IoU ≥ {IOU_LABEL_THRESHOLD})")
plt.ylim(0, 1)
plt.title(f"Mean Recall vs Number of Proposals — {EXTRACTOR}")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_folder_name}/Mean_recall_IoU_{str(IOU_LABEL_THRESHOLD)}_{EXTRACTOR}.png", bbox_inches="tight", pad_inches=0)

