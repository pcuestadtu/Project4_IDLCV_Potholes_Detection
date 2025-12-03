import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms, ops
from PIL import Image
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models

# ==========================================
# CONFIGURATION
# ==========================================
SPLITS_PATH = "./splits.json"
# Chemins basés sur tes documents [cite: 12]
IMAGE_DIR = "/dtu/datasets1/02516/potholes/images"
ANNOTATION_DIR = "/dtu/datasets1/02516/potholes/annotations"
MODEL_PATH = "./part2/best_pothole_classifier_v2.pth" 
PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Détection automatique du périphérique (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Exécution sur : {device}")

# ==========================================
# UTILITAIRES : 1. PARSING XML (PASCAL VOC)
# ==========================================
def parse_voc_xml(xml_file):
    """
    Extrait les bounding boxes de vérité terrain depuis un fichier XML.
    Retourne une liste de listes : [[x1, y1, x2, y2], ...]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    
    for obj in root.findall("object"):
        name = obj.find("name").text
        # On ne s'intéresse qu'aux nids-de-poule
        if name != "pothole":
            continue
            
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        
        boxes.append([xmin, ymin, xmax, ymax])
        
    return np.array(boxes)

# ==========================================
# UTILITAIRES : 2. EVALUATION (IOU & AP)
# ==========================================
def compute_iou(boxA, boxB):
    """Calcule l'Intersection over Union entre deux boîtes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def compute_average_precision(all_predictions, all_gts, iou_threshold=0.5):
    """
    Calcule l'Average Precision (AP) pour la classe Pothole.
    """
    true_positives = []
    scores = []
    num_gt_total = 0

    # Aplatir les listes pour le calcul global
    for i in range(len(all_predictions)):
        pred_boxes = all_predictions[i]['boxes']
        pred_scores = all_predictions[i]['scores']
        gt_boxes = all_gts[i]
        
        num_gt_total += len(gt_boxes)
        
        detected_gt = [False] * len(gt_boxes)
        
        # Pour chaque prédiction (déjà triée par score dans l'étape précédente idéalement)
        # On refait un tri local au cas où
        if len(pred_boxes) == 0:
            continue
            
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        for j, box in enumerate(pred_boxes):
            scores.append(pred_scores[j])
            
            if len(gt_boxes) == 0:
                true_positives.append(0)
                continue

            # Trouver le GT avec le meilleur IoU
            ious = [compute_iou(box, gt) for gt in gt_boxes]
            max_iou = max(ious)
            max_idx = np.argmax(ious)

            if max_iou >= iou_threshold:
                if not detected_gt[max_idx]:
                    true_positives.append(1)
                    detected_gt[max_idx] = True
                else:
                    true_positives.append(0) # Déjà détecté (Double detection)
            else:
                true_positives.append(0) # IoU trop faible

    if num_gt_total == 0:
        return 0.0, [], []

    # Conversion en numpy arrays
    scores = np.array(scores)
    true_positives = np.array(true_positives)

    # Trier par score décroissant pour la courbe PR
    indices = np.argsort(-scores)
    true_positives = true_positives[indices]

    # Cumul des TP et FP
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(1 - true_positives)

    # Calcul Précision et Rappel
    recalls = tp_cumsum / num_gt_total
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # Calcul de l'AP (Area Under Curve)
    # Ajout des points sentinelles (0, 1) et (1, 0) pour l'intégration
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Enveloppe convexe pour la précision (standard PASCAL VOC)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Intégration
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap, precisions, recalls

def plot_precision_recall_curve(precisions, recalls, output_path):
    plt.figure()
    plt.plot(recalls, precisions, label='Pothole')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# ==========================================
# UTILITAIRES : 3. SELECTIVE SEARCH
# ==========================================
def get_selective_search_boxes(image_path):
    im = cv2.imread(image_path)
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    
    proposals = []
    for (x, y, w, h) in rects:
        # Filtrage minimal pour éviter les minuscules boîtes
        if w > 20 and h > 20: 
            proposals.append([x, y, x + w, y + h])
    
    return np.array(proposals)

# ==========================================
# MAIN SCRIPT
# ==========================================
def main():
    # 1. Chargement du modèle (Step 2 Reuse)
    print("Chargement du modèle...")
    # --- NOUVEAU BLOC DE CHARGEMENT (Synchronisé avec train_classifier_2.0) ---

    model = models.resnet18()
    num_ftrs = model.fc.in_features

    # Il faut définir la couche finale comme un nn.Sequential (Dropout + Linear)
    # pour que les noms des poids correspondent (fc.1.weight)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), # Doit être présent pour le matching des clés !
        nn.Linear(num_ftrs, 2)
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    # Transformations (Normalisation ImageNet obligatoire pour ResNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Chargement des splits
    with open(SPLITS_PATH, "r") as f:
        splits = json.load(f)
    test_images = splits["test"]

    all_predictions = []
    all_gts = []

    print(f"Démarrage de l'inférence sur {len(test_images)} images de test...")

    for i, img_name in enumerate(test_images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        # On sépare le nom du fichier de son extension (.png, .jpg, etc.)
        base_name = os.path.splitext(img_name)[0]
        ann_path = os.path.join(ANNOTATION_DIR, base_name + ".xml")
        
        # --- TASK 1: PROPOSALS & CLASSIFICATION ---
        try:
            # Génération des proposals (Step 1 logic)
            proposal_boxes = get_selective_search_boxes(img_path)
        except Exception as e:
            print(f"Erreur SS sur {img_name}: {e}")
            continue

        # Limiter pour la vitesse (Top 3000)
        if len(proposal_boxes) > 2000:
            proposal_boxes = proposal_boxes[:2000]

        # Préparer les batchs
        image_pil = Image.open(img_path).convert("RGB")
        batch_inputs = []
        valid_boxes = []

        for box in proposal_boxes:
            x1, y1, x2, y2 = box
            crop = image_pil.crop((x1, y1, x2, y2))
            batch_inputs.append(transform(crop))
            valid_boxes.append(box)

        if not batch_inputs:
            continue

        # Inférence par batch (évite de saturer la mémoire si trop de boxes)
        BATCH_SIZE = 64
        num_boxes = len(batch_inputs)
        pothole_scores_list = []
        
        with torch.no_grad():
            for k in range(0, num_boxes, BATCH_SIZE):
                batch_tensor = torch.stack(batch_inputs[k:k+BATCH_SIZE]).to(device)
                outputs = model(batch_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                # On garde le score de la classe 1 (Pothole)
                pothole_scores_list.extend(probs[:, 1].cpu().numpy())

        boxes_np = np.array(valid_boxes)
        scores_np = np.array(pothole_scores_list)

        # Filtrage initial (Seuil de confiance faible pour garder du recall)
        mask = scores_np > 0.5
        filtered_boxes = boxes_np[mask]
        filtered_scores = scores_np[mask]

        # --- TASK 2: NMS (NON-MAXIMUM SUPPRESSION) ---
        final_boxes = np.array([])
        final_scores = np.array([])

        if len(filtered_boxes) > 0:
            # Conversion en Float tensor pour NMS (Correction du bug Int)
            boxes_tensor = torch.tensor(filtered_boxes, dtype=torch.float32).to(device)
            scores_tensor = torch.tensor(filtered_scores, dtype=torch.float32).to(device)
            
            # Utilisation de l'opérateur NMS optimisé de Torchvision 

            keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.3)
            
            keep_indices = keep_indices.cpu().numpy()
            final_boxes = filtered_boxes[keep_indices]
            final_scores = filtered_scores[keep_indices]

        # Récupération Vérité Terrain
        gt_boxes = parse_voc_xml(ann_path)
        
        all_predictions.append({"boxes": final_boxes, "scores": final_scores})
        all_gts.append(gt_boxes)

        if (i + 1) % 10 == 0:
            print(f"Traitement : {i + 1}/{len(test_images)} images done.")

    # --- TASK 3: EVALUATION (AP) ---
    print("\nCalcul de l'Average Precision (AP)...")
    ap, precisions, recalls = compute_average_precision(all_predictions, all_gts)
    print(f"==========================================")
    print(f"RÉSULTAT FINAL - Average Precision (AP): {ap:.4f}")
    print(f"==========================================")

    plot_path = os.path.join(PLOTS_DIR, "precision_recall_curve_quality_bestmodelv2_param2.png")
    plot_precision_recall_curve(precisions, recalls, output_path=plot_path)
    print(f"Courbe Precision-Recall sauvegardée : {plot_path}")



if __name__ == "__main__":
    main()