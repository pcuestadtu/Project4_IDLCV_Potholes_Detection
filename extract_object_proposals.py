import cv2
import os
import json
import numpy as np
from visualize import save_visualization

DATASET_DIRECTORY = "/dtu/datasets1/02516/potholes"
IMAGES_DIRECTORY = DATASET_DIRECTORY + "/images"
MAX_SIZE = 400




def selective_search(image_path, fast=True):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale_factor = MAX_SIZE / max(h, w)
    img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)


    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_rgb)

    if fast:
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()  # returns list of [x, y, w, h]

    boxes = []
    for (x, y, w, h) in rects:
        xmin = int(x / scale_factor)
        ymin = int(y / scale_factor)
        xmax = int((x + w) / scale_factor)
        ymax = int((y + h) / scale_factor)
        boxes.append([xmin, ymin, xmax, ymax])

    return boxes



def edge_boxes(image_path, max_boxes=1500):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection("model.yml.gz")
    edges = edge_detection.detectEdges(img_rgb)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    eb = cv2.ximgproc.createEdgeBoxes()
    eb.setMaxBoxes(max_boxes)

    boxes, scores = eb.getBoundingBoxes(edges, orimap)

    # Boxes are (x,y,w,h)
    boxes_min_max_axis = []
    for (x, y, w, h) in boxes:
        xmin = int(x)
        ymin = int(y)
        xmax = int(x + w)
        ymax = int(y + h)
        boxes_min_max_axis.append([xmin, ymin, xmax, ymax])

    return boxes_min_max_axis




images_filenames = sorted([f for f in os.listdir(IMAGES_DIRECTORY)])

proposal_extractors = ["SS", "EdgeBoxes"]

for extractor in proposal_extractors:
    
    folder_name = f"object_proposals_{extractor}"
    os.makedirs(folder_name, exist_ok=True)

    for i, img_filename in enumerate(images_filenames):
        print(f"{extractor}: {i}/{len(images_filenames)}")
        img_path = os.path.join(IMAGES_DIRECTORY, img_filename)
        if extractor == "SS":
            boxes= selective_search(img_path)
        else:
            boxes= edge_boxes(img_path)
        
        # Save to JSON
        output_path=os.path.join(folder_name, img_filename.replace(".png", ".json"))
        with open(output_path, "w") as f:
            json.dump(boxes, f, indent=4)

    print(f"Saved {extractor} proposals to folder: {folder_name}")

