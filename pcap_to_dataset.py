import os
import math
import random
import csv
import shutil
import numpy as np
from scapy.all import PcapReader, IP, TCP, UDP
from PIL import Image

# 1. Configuração de Diretórios e Classes
DIRETORIOS_ORIGEM = ["NonVPN-PCAPs-01", "NonVPN-PCAPs-02", "NonVPN-PCAPs-03"]
DIRETORIO_DESTINO = "dataset_imagens"
IMAGENS_POR_CLASSE = 2500
MANIFESTO_CSV = os.path.join(DIRETORIO_DESTINO, "dataset_manifest.csv")

# Mapeamento para as 4 categorias do artigo
MAPEAMENTO_CLASSES = {
    "streaming_video": ["video", "youtube", "vimeo", "netflix"],
    "voip_audio": ["audio", "voip", "spotify"],
    "file_transfer": ["ftps", "ftp", "scp", "sftp", "file"],
    "chat_text": ["chat", "icq"]
}

LIMITE_POR_FICHEIRO = 150

# 2. A Inovação: Conversão com Alinhamento Espacial Rígido (Defesa XAI)
def packet_to_aligned_image(packet, image_size=28, header_limit=54):
    max_bytes = image_size * image_size # 784
    raw_bytes = bytes(packet)
    
    # Congelar os primeiros 54 bytes estritamente para cabeçalhos IP/TCP
    header_bytes = raw_bytes[:header_limit]
    payload_bytes = raw_bytes[header_limit:]
    
    if len(header_bytes) < header_limit:
        header_bytes += b'\x00' * (header_limit - len(header_bytes))
        
    aligned_packet = header_bytes + payload_bytes
    
    if len(aligned_packet) > max_bytes:
        aligned_packet = aligned_packet[:max_bytes]
    else:
        aligned_packet += b'\x00' * (max_bytes - len(aligned_packet))
        
    img_array = np.frombuffer(aligned_packet, dtype=np.uint8)
    return img_array.reshape((image_size, image_size))


def listar_ficheiros_por_classe():
    ficheiros_por_classe = {classe: [] for classe in MAPEAMENTO_CLASSES}

    for pasta_origem in DIRETORIOS_ORIGEM:
        if not os.path.exists(pasta_origem):
            continue

        for raiz, _, ficheiros in os.walk(pasta_origem):
            for ficheiro in ficheiros:
                nome_min = ficheiro.lower()
                if not nome_min.endswith((".pcap", ".pcapng")):
                    continue

                classe_destino = None
                for classe, keywords in MAPEAMENTO_CLASSES.items():
                    if any(kw in nome_min for kw in keywords):
                        classe_destino = classe
                        break

                if classe_destino:
                    ficheiros_por_classe[classe_destino].append(os.path.join(raiz, ficheiro))

    rng = random.Random(42)
    for classe in ficheiros_por_classe:
        rng.shuffle(ficheiros_por_classe[classe])

    return ficheiros_por_classe


def preparar_pasta_destino(limpar_destino=True):
    if limpar_destino and os.path.exists(DIRETORIO_DESTINO):
        for classe in MAPEAMENTO_CLASSES.keys():
            pasta_classe = os.path.join(DIRETORIO_DESTINO, classe)
            if os.path.exists(pasta_classe):
                shutil.rmtree(pasta_classe)

    os.makedirs(DIRETORIO_DESTINO, exist_ok=True)
    for classe in MAPEAMENTO_CLASSES.keys():
        os.makedirs(os.path.join(DIRETORIO_DESTINO, classe), exist_ok=True)

# 3. O Trator de Dados (Lê tudo automaticamente)
def generate_dataset(limpar_destino=True):
    print(f"{'='*50}\n A INICIAR EXTRAÇÃO AUTOMÁTICA DE DADOS \n{'='*50}")

    preparar_pasta_destino(limpar_destino=limpar_destino)

    with open(MANIFESTO_CSV, "w", newline="", encoding="utf-8") as manifesto_file:
        manifesto_writer = csv.DictWriter(
            manifesto_file,
            fieldnames=["image_path", "class_name", "source_pcap", "packet_index", "source_directory"],
        )
        manifesto_writer.writeheader()

        contadores_por_classe = {classe: 0 for classe in MAPEAMENTO_CLASSES}
        total_imagens = 0

        ficheiros_por_classe = listar_ficheiros_por_classe()

        for classe_destino, ficheiros in ficheiros_por_classe.items():
            print(f"\nA processar classe: {classe_destino} ({len(ficheiros)} ficheiros)")

            if not ficheiros:
                continue

            limite_por_ficheiro = max(1, math.ceil(IMAGENS_POR_CLASSE / len(ficheiros)))
            limite_por_ficheiro = min(limite_por_ficheiro, LIMITE_POR_FICHEIRO)

            for caminho_completo in ficheiros:
                if contadores_por_classe[classe_destino] >= IMAGENS_POR_CLASSE:
                    break

                ficheiro = os.path.basename(caminho_completo)
                print(f"  -> A extrair {ficheiro} para [{classe_destino}]")

                try:
                    leitor = PcapReader(caminho_completo)
                except Exception as e:
                    print(f"     Erro a ler {ficheiro}: {e}")
                    continue

                count_local = 0
                try:
                    for pkt in leitor:
                        if contadores_por_classe[classe_destino] >= IMAGENS_POR_CLASSE:
                            break
                        if count_local >= limite_por_ficheiro:
                            break

                        if IP in pkt and (TCP in pkt or UDP in pkt):
                            img_matrix = packet_to_aligned_image(pkt)
                            img = Image.fromarray(img_matrix)

                            nome_base = os.path.splitext(ficheiro)[0]
                            nome_imagem = f"{nome_base}_{count_local}.png"
                            caminho_imagem = os.path.join(DIRETORIO_DESTINO, classe_destino, nome_imagem)

                            img.save(caminho_imagem)
                            manifesto_writer.writerow({
                                "image_path": caminho_imagem,
                                "class_name": classe_destino,
                                "source_pcap": nome_base,
                                "packet_index": count_local,
                                "source_directory": os.path.dirname(caminho_completo),
                            })
                            count_local += 1
                            contadores_por_classe[classe_destino] += 1
                            total_imagens += 1
                finally:
                    leitor.close()

            if all(contador >= IMAGENS_POR_CLASSE for contador in contadores_por_classe.values()):
                break
                    
    print(f"\n{'='*50}")
    print(f" EXTRAÇÃO CONCLUÍDA! Total de imagens: {total_imagens}")
    for classe, contador in contadores_por_classe.items():
        print(f"  - {classe}: {contador}")
    print(f" Tudo guardado na nova pasta '{DIRETORIO_DESTINO}'.")
    print(f" Manifesto guardado em '{MANIFESTO_CSV}'.")
    print(f"{'='*50}")

    if any(contador != IMAGENS_POR_CLASSE for contador in contadores_por_classe.values()):
        print("AVISO: Nem todas as classes atingiram a quota de 2500 imagens.")

if __name__ == "__main__":
    generate_dataset()