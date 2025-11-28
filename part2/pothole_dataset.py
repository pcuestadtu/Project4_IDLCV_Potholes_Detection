import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset


class PotholeProposalDataset(Dataset):
    def __init__(self, split_file, root_dir, proposals_dir, transform=None, balance_ratio=0.25, is_train=True):
        """
        Args:
            split_file (string): Path to splits.json.
            root_dir (string): Directory with all images.
            proposals_dir (string): Directory with labeled JSON proposals.
            transform (callable, optional): Transform to be applied on a sample.
            balance_ratio (float): The desired fraction of positive samples in the dataset 
                                   (e.g., 0.25 means 25% positives, 75% negatives).
            is_train (bool): If True, applies balancing logic.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        # Select train or val split
        image_filenames = splits['train'] if is_train else splits['val']
        
        positives = []
        negatives = []

        print(f"Loading {'training' if is_train else 'validation'} proposals...")

        for img_name in image_filenames:
            prop_filename = img_name.replace('.png', '.json')
            prop_path = os.path.join(proposals_dir, prop_filename)
            
            if not os.path.exists(prop_path):
                continue
                
            with open(prop_path, 'r') as f:
                proposals = json.load(f)
            
            for p in proposals:
                label_str = p['label']
                label_int = 1 if label_str == 'pothole' else 0
                sample = (img_name, p['box'], label_int)
                
                if label_int == 1:
                    positives.append(sample)
                else:
                    negatives.append(sample)

        # --- Class Imbalance Handling ---
        if is_train and balance_ratio is not None:
            if len(positives) == 0:
                print("Warning: No positive samples found. Dataset will be empty or only negatives.")
                self.samples = negatives
            else:
                # Calculate required negatives to satisfy: n_pos / (n_pos + n_neg) = balance_ratio
                # Derived: n_neg = n_pos * (1 - ratio) / ratio
                n_neg = int(len(positives) * (1 - balance_ratio) / balance_ratio)
                
                # Ensure we don't try to sample more negatives than exist
                n_neg = min(n_neg, len(negatives))
                
                if n_neg < len(negatives):
                    sampled_negatives = random.sample(negatives, n_neg)
                else:
                    sampled_negatives = negatives
                
                self.samples = positives + sampled_negatives
                random.shuffle(self.samples)
                
                print(f"Balanced Data (Ratio ~{balance_ratio:.2f}): {len(positives)} Positives, {len(sampled_negatives)} Negatives")
        else:
            # For validation or if ratio is None, use all data
            self.samples = positives + negatives
            print(f"Total Samples (Unbalanced): {len(self.samples)} ({len(positives)} Pos, {len(negatives)} Neg)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, box, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        # crop = image.crop((left, top, right, bottom))
        crop = image.crop((box[0], box[1], box[2], box[3]))
        
        if self.transform:
            crop = self.transform(crop)
            
        return crop, label