import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from load_dataset import build_metadata
from model_cnn import get_model
from pcap_to_dataset import IP, TCP, UDP, PcapReader, packet_to_aligned_image, listar_ficheiros_por_classe


def _get_rss_mb():
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if os.name == "posix":
                return rss / 1024.0
            return rss / (1024 * 1024)
        except Exception:
            return float("nan")


def _find_representative_pcap_paths():
    files_by_class = listar_ficheiros_por_classe()
    pcap_paths = []
    for class_name in sorted(files_by_class):
        if files_by_class[class_name]:
            pcap_paths.append(files_by_class[class_name][0])
    return pcap_paths


def _collect_packets(pcap_paths, target_count):
    packets = []
    for pcap_path in pcap_paths:
        with PcapReader(pcap_path) as reader:
            for packet in reader:
                if IP in packet and (TCP in packet or UDP in packet):
                    packets.append(packet)
                    if len(packets) >= target_count:
                        return packets
    return packets


def benchmark_pipeline(model_name, checkpoint_path, warmup=20, repeats=200, threads=1):
    dataset, _, _, _ = build_metadata()
    device = torch.device("cpu")
    torch.set_grad_enabled(False)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    model = get_model(model_name=model_name, num_classes=len(dataset.classes)).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    packet_budget = warmup + repeats
    pcap_paths = _find_representative_pcap_paths()
    packets = _collect_packets(pcap_paths, packet_budget)

    if len(packets) < packet_budget:
        raise RuntimeError(f"Apenas foram recolhidos {len(packets)} pacotes válidos; esperado {packet_budget}.")

    latencies_ms = []
    rss_before = _get_rss_mb()

    with torch.no_grad():
        for packet in packets[:warmup]:
            image_matrix = packet_to_aligned_image(packet)
            input_tensor = torch.from_numpy(image_matrix).unsqueeze(0).unsqueeze(0).float() / 255.0
            _ = model(input_tensor)

        start = time.perf_counter()
        for packet in packets[warmup:]:
            packet_start = time.perf_counter()
            image_matrix = packet_to_aligned_image(packet)
            input_tensor = torch.from_numpy(image_matrix).unsqueeze(0).unsqueeze(0).float() / 255.0
            _ = model(input_tensor)
            packet_end = time.perf_counter()
            latencies_ms.append((packet_end - packet_start) * 1000)
        end = time.perf_counter()

    rss_after = _get_rss_mb()
    total_time_seconds = end - start
    pps = repeats / total_time_seconds if total_time_seconds > 0 else float("inf")
    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    latencies_ms = np.asarray(latencies_ms, dtype=np.float64)

    print(f"{'=' * 70}")
    print(" NEUROFLOW END-TO-END EDGE BENCHMARK")
    print(f"{'=' * 70}")
    print(f"Modelo: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Pacotes válidos usados: {packet_budget}")
    print(f"Warm-up: {warmup} | Medições: {repeats}")
    print(f"Threads CPU: {threads}")
    print(f"Latência média total do pipeline: {latencies_ms.mean():.4f} ms")
    print(f"Latência mediana: {np.median(latencies_ms):.4f} ms")
    print(f"Latência p95: {np.percentile(latencies_ms, 95):.4f} ms")
    print(f"PPS (packets per second): {pps:.2f}")
    print(f"RSS antes: {rss_before:.2f} MB")
    print(f"RSS após: {rss_after:.2f} MB")
    print(f"Delta RSS: {(rss_after - rss_before):.2f} MB")
    print(f"Tamanho do modelo: {model_size_mb:.2f} MB")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark end-to-end do pipeline NeuroFlow")
    parser.add_argument("--model-name", default="custom", choices=["custom", "mobilenetv2", "resnet18"], help="Arquitetura a avaliar")
    parser.add_argument("--checkpoint", default="neuroflow_custom.pth", help="Checkpoint do modelo")
    parser.add_argument("--warmup", type=int, default=20, help="Número de pacotes para aquecimento")
    parser.add_argument("--repeats", type=int, default=200, help="Número de pacotes medidos")
    parser.add_argument("--threads", type=int, default=1, help="Threads CPU")
    args = parser.parse_args()

    benchmark_pipeline(args.model_name, args.checkpoint, warmup=args.warmup, repeats=args.repeats, threads=args.threads)