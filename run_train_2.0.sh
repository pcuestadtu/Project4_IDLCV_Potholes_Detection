#!/bin/bash
#BSUB -q c02516
#BSUB -J train_pothole_2.0
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -M 20GB
#BSUB -W 04:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

# --- COMMANDES D'EXECUTION ---
mkdir -p logs
echo "Démarrage du job avec ID \$LSB_JOBID sur hôte \$(hostname)"

# Activation de TON environnement virtuel
source env_DL/bin/activate

echo "Lancement du script d'entraînement amélioré..."
python3 part2/train_classifier_2.0.py