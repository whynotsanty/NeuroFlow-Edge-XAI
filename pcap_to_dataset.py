from scapy.all import rdpcap, IP, TCP, UDP
import numpy as np
from PIL import Image
import os

def packet_to_matrix(packet, image_size=28):
    """
    Transforma um pacote de rede numa matriz NumPy 28x28 (784 bytes)
    com a lógica de Truncatura e Zero-Padding.
    """
    max_bytes = image_size * image_size
    raw_bytes = bytes(packet)
    byte_list = list(raw_bytes)
    
    if len(byte_list) > max_bytes:
        byte_list = byte_list[:max_bytes]
    elif len(byte_list) < max_bytes:
        byte_list.extend([0] * (max_bytes - len(byte_list)))
        
    # Criar a matriz 28x28
    return np.array(byte_list, dtype=np.uint8).reshape((image_size, image_size))

def create_dataset_from_pcap(pcap_file, output_folder, max_images=100):
    """
    Lê o .pcap e guarda os primeiros 'max_images' pacotes como ficheiros PNG.
    """
    # Garantir que a pasta de destino existe
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"A processar '{pcap_file}' -> A guardar em '{output_folder}'...")
    
    try:
        # Nota: rdpcap carrega o ficheiro todo para a memória. 
        # Para ficheiros massivos no futuro usaremos PcapReader, mas para este teste serve perfeitamente.
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"Erro ao ler o ficheiro: {e}")
        return

    saved_count = 0
    
    for pkt in packets:
        if IP in pkt and (TCP in pkt or UDP in pkt):
            # 1. Obter a matriz 28x28 do NumPy
            matrix = packet_to_matrix(pkt)
            
            # 2. O Pillow (Image) transforma a matriz diretamente numa imagem em tons de cinza ('L')
            img = Image.fromarray(matrix, mode='L')
            
            # 3. Guardar no disco com um nome numerado
            filename = f"packet_{saved_count:04d}.png"
            filepath = os.path.join(output_folder, filename)
            img.save(filepath)
            
            saved_count += 1
            if saved_count >= max_images:
                break
                
    print(f"Sucesso! Geradas {saved_count} imagens em '{output_folder}'.")

if __name__ == "__main__":
    # Vamos configurar a fábrica para processar duas classes diferentes do teu print
    
    # Classe 1: Tráfego de Vídeo do Facebook
    pcap_video = "NonVPN-PCAPs-01/facebook_video1a.pcap"
    folder_video = "dataset_imagens/streaming_video"
    
    # Classe 2: Tráfego de Chat do Facebook
    pcap_chat = "NonVPN-PCAPs-01/facebook_chat_4a.pcap"
    folder_chat = "dataset_imagens/web_chat"
    
    # Executar a extração para as duas classes (limitado a 100 imagens cada para teste)
    if os.path.exists(pcap_video):
        create_dataset_from_pcap(pcap_video, folder_video, max_images=100)
    else:
        print(f"Não encontrei o ficheiro de vídeo: {pcap_video}")
        
    if os.path.exists(pcap_chat):
        create_dataset_from_pcap(pcap_chat, folder_chat, max_images=100)
    else:
        print(f"Não encontrei o ficheiro de chat: {pcap_chat}")