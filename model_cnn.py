import torch
import torch.nn as nn
from torchvision import models


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.activation(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.activation(x)
        return x


# --- 1. A NOSSA CNN CASEIRA (Compacta e Edge-Friendly) ---
class NeuroFlowCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(NeuroFlowCNN, self).__init__()
        
        # STEM: Mantém resolução 28x28 (stride=1). 
        # Fundamental para ler os primeiros 54 bytes de cabeçalho intactos.
        self.stem = ConvBNReLU(1, 16, stride=1)
        
        # BLOCO 1: Extração profunda sem reduzir o espaço
        self.block1 = DepthwiseSeparableBlock(16, 32, stride=1)
        # Primeiro downsampling (passa para 14x14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # BLOCO 2: Aprofundar as features
        self.block2 = DepthwiseSeparableBlock(32, 64, stride=1)
        # Segundo downsampling (passa para 7x7)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # CLASSIFICADOR: Substituímos o AdaptiveAvgPool por um Flatten rigoroso
        # 64 canais * 7 * 7 = 3136 features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.classifier(x)
        return x


# --- 2. O MOTOR DE SELEÇÃO DE ARQUITETURAS ---
def get_model(model_name="custom", num_classes=4):
    """
    Fábrica que devolve o modelo escolhido e já adaptado para 1 canal (escala de cinzas).
    """
    if model_name == "custom":
        print("A carregar modelo: NeuroFlowCNN compacta para Edge (1 canal, 28x28)")
        return NeuroFlowCNN(num_classes=num_classes)

    elif model_name == "resnet18":
        print("A carregar modelo: ResNet-18 (Força Bruta e Alta Precisão)")
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
    # Teste rápido da arquitetura compacta para 28x28 com as nossas 4 classes.
    modelo_escolhido = get_model(model_name="custom", num_classes=4)
    
    # Criar um lote falso de 32 pacotes em escala de cinzas (1 canal, 28x28)
    pacote_falso = torch.randn(32, 1, 28, 28)
    previsao = modelo_escolhido(pacote_falso)
    
    print("\n--- Teste de Saída da CNN Customizada ---")
    print(f"Formato da Saída: {previsao.shape}") 
    
    if previsao.shape == (32, 4):
        print("✅ Sucesso: O tensor de saída está dimensionado corretamente para o cálculo de Entropia.")
    else:
        print("❌ Erro no dimensionamento das camadas lineares.")