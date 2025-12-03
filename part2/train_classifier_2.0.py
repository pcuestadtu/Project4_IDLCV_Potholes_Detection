import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from pothole_dataset import PotholeProposalDataset
import os
import copy

# --- Hyperparameters ---
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
PATIENCE = 7 

# --- CHEMINS CORRIGÉS (Exécution depuis la racine) ---
PROPOSALS_DIR = "./labeled_object_proposals_SS" 
IMAGES_DIR = "/dtu/datasets1/02516/potholes/images"
SPLIT_FILE = "./splits.json" 

# Sauvegarde dans le dossier part2 pour que ce soit rangé
# J'ai ajouté '_v2' pour ne pas écraser ton modèle actuel pendant que tu testes
SAVE_PATH = "part2/best_pothole_classifier_v2.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Split file: {os.path.abspath(SPLIT_FILE)}")

    # --- 1. Enhanced Transforms (Data Augmentation) ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- 2. Create Datasets (AVEC PATCH DE SECOURS) ---
    print("Chargement des données...")
    
    full_train_dataset = PotholeProposalDataset(
        SPLIT_FILE, IMAGES_DIR, PROPOSALS_DIR, 
        transform=data_transforms['train'], is_train=True
    )
    
    val_check = PotholeProposalDataset(
        SPLIT_FILE, IMAGES_DIR, PROPOSALS_DIR, 
        transform=data_transforms['val'], is_train=False
    )

    if len(val_check) == 0:
        print("⚠️  ATTENTION: Dataset Validation vide. Activation du SPLIT automatique (90/10).")
        total_size = len(full_train_dataset)
        val_size = int(0.1 * total_size)
        train_size = total_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    else:
        print("Validation trouvée, utilisation normale.")
        train_dataset = full_train_dataset
        val_dataset = val_check

    print(f"Tailles finales -> Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 3. Build Model with Dropout ---
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 2)
    )
    model = model.to(device)

    # --- 4. Optimizer, Loss & Scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # --- 5. Training Loop with Early Stopping ---
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # --- VALIDATION ---
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
        
        if len(val_dataset) > 0:
            val_loss = val_running_loss / len(val_dataset)
            val_acc = val_correct / val_total
            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)

            # --- CHECKPOINTING ---
            if val_loss < best_loss:
                print(f"Validation Loss improved ({best_loss:.4f} --> {val_loss:.4f}). Saving model to {SAVE_PATH}...")
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'history': history,
                    'best_loss': best_loss
                }
                torch.save(checkpoint, SAVE_PATH)
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs.")
                
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered! Training stopped at epoch {epoch+1}")
                break
        else:
             print("Erreur critique: Validation vide.")

    print(f"Training complete. Best Validation Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()