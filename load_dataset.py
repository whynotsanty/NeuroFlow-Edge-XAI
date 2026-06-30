import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def prepare_data(data_dir="dataset_imagens", batch_size=32):
    """
    Carrega imagens organizadas em pastas (classes), converte para Tensores
    e divide em dados de Treino e Validação.
    """
    
    # 1. Definir as Transformações
    # O ImageFolder costuma ler as imagens em RGB (3 canais). 
    # Forçamos a escala de cinzentos (1 canal) e convertemos para Tensor PyTorch.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    # 2. Carregar o Dataset (O ImageFolder lê as pastas como labels)
    print(f"A ler imagens da pasta: '{data_dir}'...")
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    print(f"Classes detetadas automaticamente: {full_dataset.classes}")
    print(f"Total de imagens encontradas: {len(full_dataset)}")
    
    # 3. Dividir em Treino (80%) e Teste/Validação (20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # 4. Criar os DataLoaders (Os "empregados de mesa" que servem os dados à CNN)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, full_dataset.classes

if __name__ == "__main__":
    # Testar o nosso pipeline de dados
    train_loader, test_loader, classes = prepare_data()
    
    # Vamos cuscar o que está dentro do primeiro "lote" (batch) de treino
    for images, labels in train_loader:
        print("\n--- Informação do Batch (Lote) ---")
        print(f"Formato do Tensor de Imagens: {images.shape}")
        print(f"Formato do Tensor de Etiquetas: {labels.shape}")
        
        # O formato esperado é [32, 1, 28, 28]
        # Significa: [Tamanho do Batch, Canais de Cor, Altura, Largura]
        break # Paramos logo após o primeiro lote