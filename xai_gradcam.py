import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from load_dataset import build_metadata, get_stratified_group_kfold_splits
from model_cnn import get_model


MODEL_NAME = "custom"
CHECKPOINT_PATH = f"neuroflow_{MODEL_NAME}.pth"
N_SPLITS = 5
RANDOM_STATE = 42


def load_model(model_name, checkpoint_path, device, num_classes):
    model = get_model(model_name=model_name, num_classes=num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_target_layer(model, model_name):
    if model_name == "custom":
        return model.block3.pointwise_conv
    if model_name == "mobilenetv2":
        return model.features[-1]
    if model_name == "resnet18":
        return model.layer4[-1]
    raise ValueError(f"Modelo sem camada alvo definida: {model_name}")


def pick_validation_sample(dataset, val_indices, target_class_name=None):
    if target_class_name is None:
        sample_index = val_indices[0]
        return sample_index

    class_to_index = {name: idx for idx, name in enumerate(dataset.classes)}
    if target_class_name not in class_to_index:
        raise ValueError(f"Classe inválida: {target_class_name}. Opções: {dataset.classes}")

    target_index = class_to_index[target_class_name]
    for sample_index in val_indices:
        if dataset.targets[sample_index] == target_index:
            return sample_index

    raise RuntimeError(f"Não foi encontrada nenhuma amostra da classe '{target_class_name}' no fold de validação.")


def generate_golden_figure(model_name=MODEL_NAME, checkpoint_path=CHECKPOINT_PATH, target_class_name=None, fold_index=0):
    print("A iniciar o Mapeamento Semântico Inverso (XAI)...")

    dataset, records, targets, groups = build_metadata()
    splits = get_stratified_group_kfold_splits(targets, groups, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    _, val_indices = splits[fold_index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name=model_name, checkpoint_path=checkpoint_path, device=device, num_classes=len(dataset.classes))

    target_layers = [get_target_layer(model, model_name)]
    cam = GradCAM(model=model, target_layers=target_layers)

    sample_index = pick_validation_sample(dataset, val_indices, target_class_name=target_class_name)
    image_tensor, label_idx = dataset[sample_index]

    input_tensor = image_tensor.unsqueeze(0).to(device)
    classe_real = dataset.classes[label_idx]

    with torch.no_grad():
        output = model(input_tensor)
        previsao_idx = torch.argmax(output, dim=1).item()
        classe_prevista = dataset.classes[previsao_idx]

    targets_cam = [ClassifierOutputTarget(previsao_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets_cam)[0]

    indice_maximo = int(np.argmax(grayscale_cam))
    linha = indice_maximo // 28
    coluna = indice_maximo % 28
    byte_foco = linha * 28 + coluna
    regiao = "CABEÇALHO" if byte_foco < 54 else "PAYLOAD"

    print(f"\n{'=' * 50}")
    print(" ALERTA DE MAPEAMENTO INVERSO (XAI)")
    print(f"{'=' * 50}")
    print(f"Amostra analisada: {dataset.classes[label_idx]}")
    print(f"Classe real: {classe_real}")
    print(f"Classe prevista: {classe_prevista}")
    print(f"Pico de atenção em Linha {linha}, Coluna {coluna}.")
    print(f"Isso corresponde ao BYTE {byte_foco} do pacote original.")
    print(f"Região interpretada: {regiao} (cabeçalho reservado até ao byte 54).")
    print(f"{'=' * 50}\n")
    print(f"Fold analisado: {fold_index + 1}/{N_SPLITS}")
    print(f"Amostras totais carregadas: {len(records)}")

    img_original = image_tensor.numpy().squeeze()
    rgb_img = np.stack((img_original,) * 3, axis=-1)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(img_original, cmap="gray")
    axes[0].set_title(f"Matriz PCAP Bruta\nReal: {classe_real}")
    axes[0].axis("off")

    axes[1].imshow(cam_image)
    axes[1].set_title(f"Grad-CAM\nPrevista: {classe_prevista}")
    axes[1].axis("off")

    plt.tight_layout()
    output_path = "xai_gradcam_neuroflow.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Figura XAI guardada em: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geração de Grad-CAM para NeuroFlow")
    parser.add_argument("--model-name", default=MODEL_NAME, choices=["custom", "mobilenetv2", "resnet18"], help="Arquitetura a explicar")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Checkpoint do modelo")
    parser.add_argument("--target-class", default=None, help="Classe a selecionar na amostra de validação")
    parser.add_argument("--fold-index", type=int, default=0, help="Fold a usar para a amostra de validação")
    args = parser.parse_args()

    generate_golden_figure(model_name=args.model_name, checkpoint_path=args.checkpoint, target_class_name=args.target_class, fold_index=args.fold_index)