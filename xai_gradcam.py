import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Importar a fábrica de modelos do teu projeto
from model_cnn import get_model
from load_dataset import prepare_data

def generate_golden_figure():
    print("A iniciar o Mapeamento Semântico Inverso (XAI)...")
    
    # 1. Carregar Dados e Classes
    _, test_loader, classes = prepare_data(data_dir="dataset_imagens", batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Carregar o Modelo Campeão Edge
    modelo = get_model(model_name="mobilenetv2", num_classes=len(classes)).to(device)
    modelo.load_state_dict(torch.load("neuroflow_mobilenetv2.pth", weights_only=True))
    modelo.eval()
    
    # 3. Escolher a última camada convolucional da MobileNetV2 (onde a "magia" da decisão acontece)
    target_layers = [modelo.features[-1]]
    
    # Construir o objeto Grad-CAM
    cam = GradCAM(model=modelo, target_layers=target_layers)
    
    # 4. Extrair uma imagem de teste
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    input_tensor = images.to(device)
    classe_real = classes[labels[0].item()]
    
    # Obter a previsão da rede
    with torch.no_grad():
        output = modelo(input_tensor)
        previsao_idx = torch.argmax(output, dim=1).item()
        classe_prevista = classes[previsao_idx]
    
    # 5. Gerar o Mapa de Calor
    # Focamos na classe que o modelo previu para perceber o "porquê"
    targets = [ClassifierOutputTarget(previsao_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # --- A INOVAÇÃO: MAPEAMENTO SEMÂNTICO INVERSO ---
    # 1. Encontrar o pixel mais "quente" no mapa de calor
    indice_maximo = np.argmax(grayscale_cam)
    linha = indice_maximo // 28
    coluna = indice_maximo % 28
    
    # 2. Traduzir a coordenada de volta para o byte do pacote original
    byte_foco = (linha * 28) + coluna
    
    print(f"\n{'='*50}")
    print(f" ALERTA DE MAPEAMENTO INVERSO (XAI) ")
    print(f"{'='*50}")
    print(f"A rede focou a sua atenção máxima na Linha {linha}, Coluna {coluna}.")
    print(f"Isso corresponde ao BYTE {byte_foco} do pacote original!")
    print(f"(Assumindo um cabeçalho IP/TCP clássico de 54 bytes, o byte {byte_foco} faz parte do {'CABEÇALHO' if byte_foco < 54 else 'PAYLOAD'}).")
    print(f"{'='*50}\n")
    
    # 6. Preparar a Imagem Visual (Para o Painel A e B do Artigo)
    # Converter o tensor 28x28 (escala de cinza 1 canal) para RGB para o GradCAM conseguir desenhar por cima
    img_original = images[0].cpu().numpy().squeeze() 
    rgb_img = np.stack((img_original,)*3, axis=-1) 
    
    # Normalizar para [0, 1]
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    
    # Sobrepor o mapa de calor à imagem original
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # 7. Renderizar a "Figura de Ouro"
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Painel A: Matriz PCAP Bruta
    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title(f"Painel A: Matriz Bruta (Classe: {classe_real})")
    axes[0].axis('off')
    
    # Painel B: Mapa de Calor Grad-CAM
    axes[1].imshow(cam_image)
    axes[1].set_title(f"Painel B: Decisão XAI (Previsão: {classe_prevista})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_golden_figure()