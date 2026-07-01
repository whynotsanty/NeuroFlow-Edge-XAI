import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from load_dataset import build_metadata, get_stratified_group_kfold_splits
from model_cnn import get_model


DATA_DIR = "dataset_imagens"
MODEL_NAME = "custom"
NUM_EPOCHS = 50
N_SPLITS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
RANDOM_STATE = 42


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, indices, batch_size=BATCH_SIZE, shuffle=True):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def train_one_fold(model, train_loader, device, epochs=NUM_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        mean_loss = running_loss / max(1, len(train_loader))
        print(f"  Época [{epoch + 1:02d}/{epochs}] | Loss: {mean_loss:.4f}")


@torch.no_grad()
def evaluate_fold(model, val_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


@torch.no_grad()
def measure_latency(model, device):
    model.eval()
    sample = torch.zeros(1, 1, 28, 28, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    _ = model(sample)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000


def train_and_profile(model_name=MODEL_NAME, epochs=NUM_EPOCHS):
    set_seed()

    dataset, records, targets, groups = build_metadata()
    classes = dataset.classes
    model_save_path = f"neuroflow_{model_name}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_indices = get_stratified_group_kfold_splits(targets, groups, n_splits=N_SPLITS, random_state=RANDOM_STATE)

    fold_metrics = []
    saved_model_size_mb = None

    print(f"\n{'=' * 50}")
    print(f" A INICIAR TREINO COM STRATIFIED {N_SPLITS}-FOLD: {model_name.upper()}")
    print(f"{'=' * 50}")

    for fold_idx, (train_idx, val_idx) in enumerate(split_indices, start=1):
        print(f"\nFold {fold_idx}/{N_SPLITS}")

        train_loader = make_loader(dataset, train_idx, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_loader(dataset, val_idx, batch_size=BATCH_SIZE, shuffle=False)

        model = get_model(model_name=model_name, num_classes=len(classes)).to(device)

        start_train = time.time()
        train_one_fold(model, train_loader, device, epochs=epochs)
        fold_train_time = time.time() - start_train

        metrics = evaluate_fold(model, val_loader, device)
        fold_metrics.append(metrics)

        print(f"  Accuracy do fold: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy do fold: {metrics['balanced_accuracy']:.4f}")
        print(f"  Macro F1 do fold: {metrics['macro_f1']:.4f}")
        print(f"  Tempo de treino do fold: {fold_train_time:.2f} s")

        if fold_idx == 1:
            torch.save(model.state_dict(), model_save_path)
            saved_model_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)

    if saved_model_size_mb is None:
        raise RuntimeError("Falha ao guardar o modelo do primeiro fold.")

    mean_accuracy = float(np.mean([m["accuracy"] for m in fold_metrics]))
    std_accuracy = float(np.std([m["accuracy"] for m in fold_metrics], ddof=1)) if len(fold_metrics) > 1 else 0.0
    mean_balanced_accuracy = float(np.mean([m["balanced_accuracy"] for m in fold_metrics]))
    mean_macro_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics]))

    report_model = get_model(model_name=model_name, num_classes=len(classes)).to(device)
    report_model.load_state_dict(torch.load(model_save_path, map_location=device))
    latency_ms = measure_latency(report_model, device)

    print(f"\n{'=' * 50}")
    print(" RELATÓRIO PARA O COMITÉ TÉCNICO")
    print(f"{'=' * 50}")
    print(f"Precisão média global do {N_SPLITS}-Fold: {mean_accuracy * 100:.2f}% (+/- {std_accuracy * 100:.2f}%)")
    print(f"Balanced Accuracy média: {mean_balanced_accuracy * 100:.2f}%")
    print(f"Macro F1 médio: {mean_macro_f1 * 100:.2f}%")
    print(f"Tamanho do modelo .pth: {saved_model_size_mb:.2f} MB")
    print(f"Latência de inferência (1 pacote, 1x1x28x28): {latency_ms:.4f} ms")
    print(f"Classes: {classes}")
    print(f"Total de amostras: {len(records)}")
    print(f"Modelo guardado para XAI em: {model_save_path}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino da NeuroFlow CNN compacta com validação cruzada estratificada por grupo")
    parser.add_argument("--model-name", default=MODEL_NAME, choices=["custom", "resnet18", "mobilenetv2"], help="Arquitetura a treinar")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Número de épocas por fold")
    args = parser.parse_args()

    train_and_profile(model_name=args.model_name, epochs=args.epochs)