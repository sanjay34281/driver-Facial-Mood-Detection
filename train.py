import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, f1_score

# Add root path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from models.efficientnet_classifier import DriverStateClassifier
from src.dataset_utils import get_dataloaders, check_dataset_status, generate_synthetic_starter_data

def train_driver_model(epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE):
    print("=" * 65)
    print("     EFFICIENTNET-B0 DRIVER STATE CLASSIFIER TRAINING")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # 1. Dataset Health Check
    stats = check_dataset_status()
    total_train = sum(stats["train"].values())
    total_val = sum(stats["val"].values())
    print(f"[Dataset] Train samples: {total_train} {stats['train']}")
    print(f"[Dataset] Val samples:   {total_val} {stats['val']}")

    if total_train == 0 or any(count == 0 for count in stats["train"].values()):
        print("\n[Notice] Incomplete training dataset detected in dataset/train/.")
        print("[Notice] Generating synthetic starter data so you can test the full pipeline...")
        generate_synthetic_starter_data(samples_per_class=20)
        stats = check_dataset_status()

    # 2. Data Loaders
    train_loader, val_loader, class_names = get_dataloaders(batch_size=batch_size)
    print(f"[Dataset] Classes identified: {class_names}")

    # 3. Model Setup
    model = DriverStateClassifier(num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # 4. Criterion, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # History tracking
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": []
    }

    best_val_acc = 0.0
    best_weights_path = config.CLASSIFIER_WEIGHTS_PATH
    start_time = time.time()

    print("\n[Training] Starting Transfer Learning Pipeline...")
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train_samples += inputs.size(0)

        epoch_train_loss = running_loss / total_train_samples
        epoch_train_acc = correct_train / total_train_samples

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val_samples = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val_samples += inputs.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / total_val_samples
        epoch_val_acc = correct_val / total_val_samples
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average="weighted", zero_division=0)

        scheduler.step()

        # Record history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_f1"].append(epoch_val_f1)

        # Print progress
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc*100:5.2f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc*100:5.2f}% | F1: {epoch_val_f1:.4f}")

        # Checkpoint Best Model
        if epoch_val_acc > best_val_acc or epoch == 1:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": epoch_val_acc,
                "val_f1": epoch_val_f1,
                "classes": class_names
            }, best_weights_path)
            print(f"  --> Checkpoint saved! New Best Validation Accuracy: {best_val_acc*100:.2f}%")

    elapsed = time.time() - start_time
    print(f"\n[Training Complete] Total time: {elapsed/60:.2f} mins. Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"[Checkpoint] Weights saved to: {best_weights_path}")

    # 5. Plot and save training history
    plot_path = config.BASE_DIR / "training_curves.png"
    plot_training_curves(history, plot_path)
    print(f"[Analytics] Training curves saved to {plot_path}")

    return model, history


def plot_training_curves(history, save_path):
    """Generates and saves Loss & Accuracy curve charts."""
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    plt.plot(history["val_loss"], label="Val Loss", color="crimson", lw=2)
    plt.title("EfficientNet-B0 Loss Curves", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot([acc * 100 for acc in history["train_acc"]], label="Train Acc", color="royalblue", lw=2)
    plt.plot([acc * 100 for acc in history["val_acc"]], label="Val Acc", color="crimson", lw=2)
    plt.title("Accuracy Curves (%)", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    train_driver_model()
