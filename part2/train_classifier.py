import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from pothole_dataset import PotholeProposalDataset
import os

# --- Hyperparameters ---
BATCH_SIZE = 32
NUM_EPOCHS = 10  # Can likely converge in fewer epochs
LEARNING_RATE = 0.001
PROPOSALS_DIR = "../labeled_object_proposals_SS" # or EdgeBoxes
IMAGES_DIR = "/dtu/datasets1/02516/potholes/images"
SPLIT_FILE = "../splits.json"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Transforms (Preprocessing) ---
    # R-CNN warps proposals to fixed size (e.g. 224x224)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- 2. Create Datasets ---
    # Note: is_train=True enables class balancing
    train_dataset = PotholeProposalDataset(
        SPLIT_FILE, IMAGES_DIR, PROPOSALS_DIR, 
        transform=data_transforms['train'], is_train=True
    )
    
    val_dataset = PotholeProposalDataset(
        SPLIT_FILE, IMAGES_DIR, PROPOSALS_DIR, 
        transform=data_transforms['val'], is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 3. Build Model ---
    # Using ResNet18 as a standard, efficient baseline (Slides mention VGG/AlexNet, ResNet is standard modern equiv)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze backbone (optional, but good for small datasets)
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # Replace final layer for 2 classes (Background vs Pothole)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model = model.to(device)

    # --- 4. Optimizer & Loss ---
    criterion = nn.CrossEntropyLoss()
    # If fine-tuning all layers, use small LR
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # --- 5. Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- 6. Validation Loop ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_dataset)
        val_acc = val_correct / val_total
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # --- 7. Save Model ---
    torch.save(model.state_dict(), "pothole_classifier.pth")
    print("Model saved as pothole_classifier.pth")

if __name__ == "__main__":
    main()