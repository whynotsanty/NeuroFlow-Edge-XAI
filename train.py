import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# Importar as tuas ferramentas modulares
from load_dataset import prepare_data
from model_cnn import get_model

def train_and_profile(model_name="custom", epochs=5):
    # 1. Carregar os Dados
    train_loader, test_loader, classes = prepare_data(data_dir="dataset_imagens", batch_size=16)
    
    # 2. Configurar o Motor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chamar o Seletor de Modelos!
    modelo = get_model(model_name=model_name, num_classes=len(classes)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelo.parameters(), lr=0.001)
    
    # 3. O Ciclo de Treino
    print(f"\n{'='*50}")
    print(f" A INICIAR TREINO E PERFILAMENTO: {model_name.upper()}")
    print(f"{'='*50}")
    
    start_train_time = time.time()
    
    for epoch in range(epochs):
        modelo.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = modelo(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Época [{epoch+1}/{epochs}] | Erro (Loss): {running_loss/len(train_loader):.4f}")
        
    total_train_time = time.time() - start_train_time
    print(f"Treino concluído em {total_train_time:.2f} segundos.")
    
    # 4. Guardar o Modelo e Medir o Peso (MB)
    save_path = f"neuroflow_{model_name}.pth"
    torch.save(modelo.state_dict(), save_path)
    tamanho_mb = os.path.getsize(save_path) / (1024 * 1024)
    
    # 5. O Teste de Velocidade Edge (Latência/Inferência)
    print("\n--- Métricas para o Artigo Científico ---")
    modelo.eval()
    
    # Simular a chegada de 1 único pacote de rede ao Router
    pacote_teste = torch.randn(1, 1, 28, 28).to(device)
    
    with torch.no_grad():
        # "Warm-up" (A primeira passagem no PyTorch aloca memória e é sempre lenta, por isso ignoramos)
        _ = modelo(pacote_teste)
        
        # Medir a sério
        start_infer = time.time()
        _ = modelo(pacote_teste)
        end_infer = time.time()
        
    tempo_ms = (end_infer - start_infer) * 1000
    
    print(f"1. Tamanho do Cérebro: {tamanho_mb:.2f} MB")
    print(f"2. Tempo de Decisão (1 pacote): {tempo_ms:.4f} ms")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # O Torneio de Inteligência Artificial do NeuroFlow!
    # Vamos treinar e cronometrar os três modelos consecutivamente.
    
    modelos_para_testar = ["custom", "mobilenetv2", "resnet18"]
    
    for m in modelos_para_testar:
        # Mantemos apenas 5 épocas para ser rápido.
        # Numa fase final de projeto, subíamos para 50 ou 100.
        train_and_profile(model_name=m, epochs=5)