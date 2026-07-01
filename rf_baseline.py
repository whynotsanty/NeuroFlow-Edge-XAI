import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from load_dataset import build_metadata, get_stratified_group_kfold_splits


DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42


def load_feature_matrix(dataset):
    features = []
    labels = []

    for index in range(len(dataset)):
        image_tensor, label = dataset[index]
        features.append(image_tensor.numpy().reshape(-1))
        labels.append(label)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def evaluate_random_forest(n_estimators=150, max_depth=None, n_splits=DEFAULT_N_SPLITS, random_state=DEFAULT_RANDOM_STATE):
    dataset, records, targets, groups = build_metadata()
    x, y = load_feature_matrix(dataset)

    splits = get_stratified_group_kfold_splits(targets, groups, n_splits=n_splits, random_state=random_state)

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

        classifier.fit(x[train_idx], y[train_idx])
        predictions = classifier.predict(x[val_idx])

        metrics = {
            "accuracy": accuracy_score(y[val_idx], predictions),
            "balanced_accuracy": balanced_accuracy_score(y[val_idx], predictions),
            "macro_f1": f1_score(y[val_idx], predictions, average="macro", zero_division=0),
        }
        fold_metrics.append(metrics)

        print(f"Fold {fold_idx}/{n_splits}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")

    mean_accuracy = float(np.mean([metric["accuracy"] for metric in fold_metrics]))
    mean_balanced_accuracy = float(np.mean([metric["balanced_accuracy"] for metric in fold_metrics]))
    mean_macro_f1 = float(np.mean([metric["macro_f1"] for metric in fold_metrics]))

    print(f"\n{'=' * 50}")
    print(" BASELINE RANDOM FOREST")
    print(f"{'=' * 50}")
    print(f"Amostras: {len(records)}")
    print(f"Precisão média: {mean_accuracy * 100:.2f}%")
    print(f"Balanced Accuracy média: {mean_balanced_accuracy * 100:.2f}%")
    print(f"Macro F1 médio: {mean_macro_f1 * 100:.2f}%")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Random Forest para NeuroFlow")
    parser.add_argument("--n-estimators", type=int, default=150, help="Número de árvores")
    parser.add_argument("--max-depth", type=int, default=None, help="Profundidade máxima das árvores")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS, help="Número de folds")
    args = parser.parse_args()

    evaluate_random_forest(n_estimators=args.n_estimators, max_depth=args.max_depth, n_splits=args.n_splits)