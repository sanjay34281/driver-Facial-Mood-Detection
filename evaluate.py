import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Add root path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from models.efficientnet_classifier import get_classifier
from src.dataset_utils import get_dataloaders

def evaluate_model(weights_path=config.CLASSIFIER_WEIGHTS_PATH, split="val"):
    print("=" * 65)
    print("      DRIVER STATE CLASSIFIER - MODEL EVALUATION")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Evaluating on {device}")

    eval_dir = config.TEST_DIR if split == "test" else config.VAL_DIR
    _, eval_loader, class_names = get_dataloaders(val_dir=eval_dir, batch_size=config.BATCH_SIZE)

    model = get_classifier(num_classes=len(class_names), weights_path=weights_path, device=device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Generate Metrics Report
    print("\n" + "=" * 65)
    print(f"CLASSIFICATION REPORT ({split.upper()} SET)")
    print("=" * 65)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    save_cm_plot(cm, class_names, config.BASE_DIR / "confusion_matrix.png")
    print(f"[Success] Confusion matrix plot saved to: {config.BASE_DIR / 'confusion_matrix.png'}")

def save_cm_plot(cm, class_names, save_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Driver State Confusion Matrix", fontsize=14)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    # Label counts inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12, fontweight="bold")

    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    evaluate_model()
