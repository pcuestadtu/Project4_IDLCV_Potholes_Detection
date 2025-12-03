import torch
import matplotlib.pyplot as plt
import os

# Assure-toi que le chemin d'accès au meilleur modèle est correct
SAVE_PATH = "part2/best_pothole_classifier_v2.pth" 
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

try:
    # 1. Charger le fichier de checkpoint
    # On charge sur CPU pour éviter les problèmes de GPU
    checkpoint = torch.load(SAVE_PATH, map_location='cpu')
    history = checkpoint['history']
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # --- GRAPHIQUE 1 : PERTE (LOSS) ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # --- GRAPHIQUE 2 : PRÉCISION (ACCURACY) ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Sauvegarder la figure pour ton rapport
    plot_file = os.path.join(PLOT_DIR, 'training_history_V2.png')
    plt.savefig(plot_file)
    print(f"\nGraphiques sauvegardés sous : {plot_file}")
    
except FileNotFoundError:
    print(f"\nERREUR: Le fichier de modèle {SAVE_PATH} n'a pas été trouvé. Le training a-t-il terminé ?")
except KeyError:
    print("\nERREUR: La clé 'history' n'a pas été trouvée dans le checkpoint. Vérifie le script de sauvegarde.")
except Exception as e:
    print(f"\nUne erreur est survenue : {e}")