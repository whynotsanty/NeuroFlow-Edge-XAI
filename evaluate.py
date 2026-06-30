import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Importar as tuas ferramentas
from load_dataset import prepare_data
from model_cnn import NeuroFlowCNN

def evaluate_model():
    # 1. Carregar os Dados de Teste (Os 20% que o modelo nunca viu!)
    # Como não precisamos de treinar, carregamos apenas a ponte
    _, test_loader, classes = prepare_data(data_dir="dataset_imagens", batch_size=16)
    
    # 2. Ligar o Motor e Carregar o Cérebro
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = NeuroFlowCNN(num_classes=len(classes)).to(device)
    
    try:
        # Carregamos os pesos (a inteligência) do ficheiro que salvaste
        modelo.load_state_dict(torch.load("neuroflow_brain.pth", weights_only=True))
        print("\nPesos 'neuroflow_brain.pth' carregados com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    # AVISO CRÍTICO: Desligar o modo de treino. 
    # Isto congela o modelo para ele não aprender com o teste.
    modelo.eval() 

    all_preds = []
    all_labels = []

    print("A avaliar as imagens de teste...\n")
    
    # torch.no_grad() desliga a máquina de derivadas do PyTorch, poupando RAM e CPU
    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # O modelo dá os palpites
            outputs = modelo(images)
            
            # Escolhemos a classe com maior probabilidade
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Imprimir o Relatório Científico
    print("=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 4. Desenhar a Matriz de Confusão Visual
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão - NeuroFlow CNN')
    plt.ylabel('Classe Real (Verdadeira)')
    plt.xlabel('Previsão do Modelo')
    plt.show()

if __name__ == "__main__":
    evaluate_model()