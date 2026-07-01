import csv
import os
import re

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DEFAULT_DATA_DIR = "dataset_imagens"
DEFAULT_RANDOM_STATE = 42
DEFAULT_BATCH_SIZE = 32
DEFAULT_MANIFEST_NAME = "dataset_manifest.csv"


def get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])


def build_dataset(data_dir=DEFAULT_DATA_DIR, transform=None):
    if transform is None:
        transform = get_transform()

    print(f"A ler imagens da pasta: '{data_dir}'...")
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"Classes detetadas automaticamente: {dataset.classes}")
    print(f"Total de imagens encontradas: {len(dataset)}")
    return dataset


def make_loader(dataset, indices, batch_size=DEFAULT_BATCH_SIZE, shuffle=False):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def _manifest_path(data_dir):
    return os.path.join(data_dir, DEFAULT_MANIFEST_NAME)


def _load_manifest_map(data_dir):
    manifest_file = _manifest_path(data_dir)
    manifest_map = {}

    if not os.path.exists(manifest_file):
        return manifest_map

    with open(manifest_file, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = os.path.normpath(row["image_path"])
            manifest_map[image_path] = {
                "class_name": row["class_name"],
                "source_pcap": row["source_pcap"],
                "packet_index": int(row["packet_index"]),
                "source_directory": row.get("source_directory", ""),
            }

    return manifest_map


def _extract_source_pcap_from_filename(image_path):
    filename = os.path.basename(image_path)
    stem, _ = os.path.splitext(filename)
    return re.sub(r"_[0-9]+$", "", stem)


def build_metadata(data_dir=DEFAULT_DATA_DIR):
    dataset = build_dataset(data_dir=data_dir)
    manifest_map = _load_manifest_map(data_dir)

    records = []
    for image_path, target in dataset.samples:
        normalized_path = os.path.normpath(image_path)
        class_name = dataset.classes[target]

        manifest_row = manifest_map.get(normalized_path)
        if manifest_row is not None:
            source_pcap = manifest_row["source_pcap"]
            packet_index = manifest_row["packet_index"]
            source_directory = manifest_row.get("source_directory", "")
        else:
            source_pcap = _extract_source_pcap_from_filename(normalized_path)
            packet_index = None
            source_directory = os.path.dirname(normalized_path)

        group_id = f"{class_name}::{source_pcap}"
        records.append(
            {
                "image_path": normalized_path,
                "class_index": target,
                "class_name": class_name,
                "source_pcap": source_pcap,
                "packet_index": packet_index,
                "source_directory": source_directory,
                "group_id": group_id,
            }
        )

    targets = np.array([record["class_index"] for record in records])
    groups = np.array([record["group_id"] for record in records])
    return dataset, records, targets, groups


def get_stratified_group_kfold_splits(targets, groups, n_splits=5, random_state=DEFAULT_RANDOM_STATE):
    indices = np.arange(len(targets))

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(indices, targets, groups))
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(indices, targets, groups))


def get_stratified_group_holdout_split(targets, groups, test_size=0.2, random_state=DEFAULT_RANDOM_STATE):
    """
    Split holdout estratificado por grupo. Se StratifiedGroupKFold existir, usa o primeiro fold.
    Caso contrário, faz uma seleção determinística aproximada por grupos.
    """

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=max(2, int(round(1 / test_size))), shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(np.zeros(len(targets)), targets, groups))
        return train_idx, test_idx
    except Exception:
        unique_groups = np.unique(groups)
        group_to_indices = {group: np.where(groups == group)[0] for group in unique_groups}
        rng = np.random.default_rng(random_state)
        shuffled_groups = list(unique_groups)
        rng.shuffle(shuffled_groups)

        test_indices = []
        seen_labels = {}
        target_test_count = max(1, int(len(targets) * test_size))

        for group in shuffled_groups:
            candidate = group_to_indices[group]
            if len(test_indices) + len(candidate) <= target_test_count:
                test_indices.extend(candidate.tolist())

        test_indices = np.array(sorted(set(test_indices)))
        train_indices = np.array(sorted(set(np.arange(len(targets))) - set(test_indices.tolist())))
        return train_indices, test_indices


def prepare_data(data_dir=DEFAULT_DATA_DIR, batch_size=DEFAULT_BATCH_SIZE, test_size=0.2, random_state=DEFAULT_RANDOM_STATE):
    dataset, _, targets, groups = build_metadata(data_dir=data_dir)
    train_idx, test_idx = get_stratified_group_holdout_split(targets, groups, test_size=test_size, random_state=random_state)

    train_loader = make_loader(dataset, train_idx, batch_size=batch_size, shuffle=True)
    test_loader = make_loader(dataset, test_idx, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, dataset.classes


if __name__ == "__main__":
    train_loader, test_loader, classes = prepare_data()

    for images, labels in train_loader:
        print("\n--- Informação do Batch (Lote) ---")
        print(f"Formato do Tensor de Imagens: {images.shape}")
        print(f"Formato do Tensor de Etiquetas: {labels.shape}")
        break