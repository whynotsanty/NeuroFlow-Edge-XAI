import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from load_dataset import build_metadata, get_stratified_group_kfold_splits, make_loader
from model_cnn import get_model


MODEL_NAME = "custom"
CHECKPOINT_PATH = f"neuroflow_{MODEL_NAME}.pth"
N_SPLITS = 5
RANDOM_STATE = 42
BATCH_SIZE = 16
CONFUSION_MATRIX_PATH = "evaluation_confusion_matrix.png"


def load_trained_model(model_name, num_classes, checkpoint_path, device):
    model = get_model(model_name=model_name, num_classes=num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model_name=MODEL_NAME, checkpoint_path=CHECKPOINT_PATH):
    dataset, records, targets, groups = build_metadata()
    classes = dataset.classes

    splits = get_stratified_group_kfold_splits(targets, groups, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    _, val_idx = splits[0]
    val_loader = make_loader(dataset, val_idx, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_name=model_name, num_classes=len(classes), checkpoint_path=checkpoint_path, device=device)

    all_preds = []
    all_labels = []

    print("A avaliar o modelo no fold de validação correspondente ao checkpoint guardado...\n")

    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_preds.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print("=== RELATÓRIO CIENTÍFICO DE AVALIAÇÃO ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4, zero_division=0))
    print(f"Amostras avaliadas: {len(records)} | Fold de validação: {len(val_idx)}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Matriz de Confusão - NeuroFlow CNN Compacta")
    plt.ylabel("Classe Real")
    plt.xlabel("Previsão")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Matriz de confusão guardada em: {os.path.abspath(CONFUSION_MATRIX_PATH)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação científica do modelo NeuroFlow")
    parser.add_argument("--model-name", default=MODEL_NAME, choices=["custom", "mobilenetv2", "resnet18"], help="Arquitetura avaliada")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Checkpoint a avaliar")
    args = parser.parse_args()

    evaluate_model(model_name=args.model_name, checkpoint_path=args.checkpoint)