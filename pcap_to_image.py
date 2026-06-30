from scapy.all import rdpcap, IP, TCP, UDP
import numpy as np
import matplotlib.pyplot as plt
import os

def packet_to_image(packet, image_size=28):
    """
    Converte um único pacote de rede numa matriz/imagem bidimensional (ex: 28x28).
    """
    max_bytes = image_size * image_size
    
    # Extrair os bytes brutos do pacote (cabeçalho + payload)
    raw_bytes = bytes(packet)
    
    # Converter para uma lista de inteiros (valores entre 0 e 255)
    byte_list = list(raw_bytes)
    
    # Tratar o tamanho do pacote (Truncar ou fazer Zero-Padding)
    if len(byte_list) > max_bytes:
        byte_list = byte_list[:max_bytes] # Corta o excesso
    elif len(byte_list) < max_bytes:
        byte_list.extend([0] * (max_bytes - len(byte_list))) # Preenche com zeros
        
    # Converter para array NumPy e fazer o reshape para 28x28
    image_matrix = np.array(byte_list, dtype=np.uint8).reshape((image_size, image_size))
    
    return image_matrix

def process_pcap(pcap_file, num_packets=3):
    """
    Lê um ficheiro .pcap e extrai as imagens dos primeiros N pacotes TCP/UDP.
    """
    print(f"\nA ler o ficheiro: {pcap_file}...")
    
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"Erro ao ler o ficheiro: {e}")
        return
    
    valid_packets = 0
    
    for pkt in packets:
        # Focamos apenas em tráfego IP que seja TCP ou UDP
        if IP in pkt and (TCP in pkt or UDP in pkt):
            img_matrix = packet_to_image(pkt)
            
            # Renderizar a imagem gerada
            plt.figure(figsize=(4, 4))
            plt.imshow(img_matrix, cmap='gray')
            plt.title(f"Pacote {valid_packets + 1} | Tamanho Original: {len(pkt)} bytes")
            plt.axis('off')
            plt.show() # O código pausa aqui até fechares a janela da imagem
            
            valid_packets += 1
            if valid_packets >= num_packets:
                break
                
    print("\nProcessamento concluído.")

if __name__ == "__main__":
    # Apontar para um ficheiro que vimos no teu print (Tráfego de Vídeo do Facebook)
    # Usamos um .pcap puro para evitar problemas de formatação inicial
    sample_pcap = "NonVPN-PCAPs-01/facebook_video1a.pcap" 
    
    if os.path.exists(sample_pcap):
        # Vamos processar os primeiros 3 pacotes válidos
        process_pcap(sample_pcap, num_packets=3)
    else:
        print(f"Ficheiro não encontrado no caminho: {sample_pcap}")
        print("Verifica se o nome da pasta e do ficheiro estão corretos.")