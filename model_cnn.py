import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --- 1. A NOSSA CNN CASEIRA (Baseline) ---
class NeuroFlowCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(NeuroFlowCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. O MOTOR DE SELEÇÃO DE ARQUITETURAS ---
def get_model(model_name="custom", num_classes=2):
    """
    Fábrica que devolve o modelo escolhido e já adaptado para 1 canal (escala de cinzas).
    """
    if model_name == "custom":
        print("A carregar modelo: Custom CNN (Baseline Leve)")
        return NeuroFlowCNN(num_classes=num_classes)

    elif model_name == "resnet18":
        print("A carregar modelo: ResNet-18 (Força Bruta e Alta Precisão)")
        # Carregamos a arquitetura sem pesos pré-treinados
        model = models.resnet18(weights=None)
        # HACK: Modificar a primeira camada para aceitar 1 canal em vez de 3
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # HACK: Modificar a última camada para as nossas N classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    elif model_name == "mobilenetv2":
        print("A carregar modelo: MobileNetV2 (Especialista em Edge Computing)")
        model = models.mobilenet_v2(weights=None)
        # HACK: Modificar a primeira camada para aceitar 1 canal
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # HACK: Modificar a última camada
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model

    else:
        raise ValueError("Modelo não reconhecido. Opções válidas: 'custom', 'resnet18', 'mobilenetv2'")

if __name__ == "__main__":
    # Teste Rápido: Podes trocar "mobilenetv2" por "resnet18" ou "custom" para ver a estrutura!
    modelo_escolhido = get_model(model_name="mobilenetv2", num_classes=2)
    
    # Criar um pacote falso para garantir que a matemática dos tensores não quebra
    pacote_falso = torch.randn(32, 1, 28, 28)
    previsao = modelo_escolhido(pacote_falso)
    
    print("\n--- Teste de Saída ---")
    print(f"Formato da Saída: {previsao.shape}") 
    # Tem de dar smp [32, 2]