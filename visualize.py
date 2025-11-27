import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import os



DATASET_DIRECTORY="/dtu/datasets1/02516/potholes" 
ANNOTATIONS_DIRECTORY=DATASET_DIRECTORY+"/annotations"
IMAGES_DIRECTORY=DATASET_DIRECTORY+"/images"


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

def save_visualization(image_dir, filename, boxes, output_path):
    # Load image
    img_path = os.path.join(image_dir, filename)
    img = Image.open(img_path)

    # Create plot
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw each bounding box
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=1, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    plt.axis("off")

    # Save figure
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)   # Important: close figure so Python doesn't use memory

    
if __name__ == "__main__":
    name, boxes = read_content(ANNOTATIONS_DIRECTORY+"/potholes0.xml")

    output_file = f"annotated_{name}"
    save_visualization(IMAGES_DIRECTORY, name, boxes, output_file)

    print("holaaaa")
    print(boxes)

    # Visualization of the rectangles extracted by SS
    # number_rects= 1000
    # output_file = f"SS_potholes{str(number_rects)}_rectangles.png"
    # print(len(img_rects))
    # save_visualization(IMAGES_DIRECTORY, name, img_rects[:number_rects] ,output_file )

