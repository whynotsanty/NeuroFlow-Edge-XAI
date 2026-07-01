import argparse
import os
from collections import Counter

import numpy as np
import torch

from load_dataset import build_metadata, get_stratified_group_kfold_splits
from model_cnn import get_model


def check_dataset_distribution(records, expected_per_class=None):
    class_counts = Counter(record["class_name"] for record in records)
    print("Distribuição por classe:")
    for class_name in sorted(class_counts):
        print(f"  - {class_name}: {class_counts[class_name]}")

    if expected_per_class is not None:
        mismatches = {
            class_name: count
            for class_name, count in class_counts.items()
            if count != expected_per_class
        }
        if mismatches:
            raise RuntimeError(f"Distribuição inválida para quota {expected_per_class}: {mismatches}")


def check_manifest(records):
    missing_paths = [r["image_path"] for r in records if not os.path.exists(r["image_path"])]
    if missing_paths:
        raise RuntimeError(f"Manifesto inconsistente: {len(missing_paths)} imagens não existem em disco.")

    empty_sources = [r for r in records if not r["source_pcap"]]
    if empty_sources:
        raise RuntimeError("Manifesto inconsistente: existem registos sem source_pcap.")

    print("Manifesto: OK (paths e source_pcap válidos)")


def check_group_leakage(targets, groups, n_splits, random_state):
    splits = get_stratified_group_kfold_splits(targets, groups, n_splits=n_splits, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise RuntimeError(f"Leakage de grupos no fold {fold_idx}: {len(overlap)} grupos em comum.")

        train_counts = Counter(targets[train_idx].tolist())
        val_counts = Counter(targets[val_idx].tolist())
        print(f"Fold {fold_idx}: train={len(train_idx)} | val={len(val_idx)} | grupos train={len(train_groups)} | grupos val={len(val_groups)}")
        print(f"  Distribuição train: {dict(sorted(train_counts.items()))}")
        print(f"  Distribuição val: {dict(sorted(val_counts.items()))}")

    print("Splits estratificados por grupo: OK (sem leakage entre train/val)")


def check_checkpoint_smoke(model_name, checkpoint_path, num_classes):
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Checkpoint não encontrado: {checkpoint_path}")

    device = torch.device("cpu")
    model = get_model(model_name=model_name, num_classes=num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    sample = torch.zeros(1, 1, 28, 28, device=device)
    with torch.no_grad():
        output = model(sample)

    if output.shape != (1, num_classes):
        raise RuntimeError(f"Saída inesperada do modelo: {tuple(output.shape)}")

    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Checkpoint: OK ({checkpoint_path}, {model_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Preflight checks do NeuroFlow antes de deployment no Raspberry Pi")
    parser.add_argument("--data-dir", default="dataset_imagens", help="Diretório do dataset de imagens")
    parser.add_argument("--model-name", default="mobilenetv2", choices=["custom", "mobilenetv2", "resnet18"], help="Arquitetura para smoke test")
    parser.add_argument("--checkpoint", default="neuroflow_mobilenetv2.pth", help="Checkpoint para validação")
    parser.add_argument("--n-splits", type=int, default=5, help="Número de folds para validar leakage")
    parser.add_argument("--random-state", type=int, default=42, help="Seed de splits")
    parser.add_argument("--expected-per-class", type=int, default=2500, help="Quota esperada por classe")
    args = parser.parse_args()

    print("=" * 70)
    print("NEUROFLOW PREFLIGHT CHECK")
    print("=" * 70)

    dataset, records, targets, groups = build_metadata(data_dir=args.data_dir)
    print(f"Total de registos: {len(records)}")

    check_dataset_distribution(records, expected_per_class=args.expected_per_class)
    check_manifest(records)
    check_group_leakage(targets, groups, n_splits=args.n_splits, random_state=args.random_state)
    check_checkpoint_smoke(args.model_name, args.checkpoint, num_classes=len(dataset.classes))

    print("=" * 70)
    print("STATUS: PRONTO PARA TESTES NO RASPBERRY PI")
    print("=" * 70)


if __name__ == "__main__":
    main()