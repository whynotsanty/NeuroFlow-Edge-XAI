import argparse
import os
import time

import numpy as np
import torch

from load_dataset import build_dataset
from model_cnn import get_model


def load_sample_tensor(data_dir):
    dataset = build_dataset(data_dir=data_dir)
    sample_tensor, sample_label = dataset[0]
    return sample_tensor.unsqueeze(0), dataset.classes, sample_label


def measure_inference_latency(model, sample_tensor, warmup=20, repeats=200):
    latencies_ms = []

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_tensor)

        for _ in range(repeats):
            start = time.perf_counter()
            _ = model(sample_tensor)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)

    return np.array(latencies_ms, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="NeuroFlow Edge benchmark for Raspberry Pi / CPU-only inference")
    parser.add_argument("--data-dir", default="dataset_imagens", help="Dataset directory used to load a representative sample")
    parser.add_argument("--checkpoint", default="neuroflow_custom.pth", help="Path to the trained checkpoint")
    parser.add_argument("--model-name", default="custom", choices=["custom", "mobilenetv2", "resnet18"], help="Model architecture to benchmark")
    parser.add_argument("--warmup", type=int, default=20, help="Warm-up iterations before timing")
    parser.add_argument("--repeats", type=int, default=200, help="Timed inference iterations")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads to use")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint não encontrado: {args.checkpoint}")

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    device = torch.device("cpu")
    sample_tensor, classes, sample_label = load_sample_tensor(args.data_dir)
    sample_tensor = sample_tensor.to(device)

    model = get_model(model_name=args.model_name, num_classes=len(classes)).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    latencies_ms = measure_inference_latency(model, sample_tensor, warmup=args.warmup, repeats=args.repeats)

    with torch.no_grad():
        output = model(sample_tensor)
        predicted_idx = int(torch.argmax(output, dim=1).item())

    model_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)

    print(f"{'=' * 60}")
    print(" NEUROFLOW EDGE BENCHMARK")
    print(f"{'=' * 60}")
    print(f"Modelo: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes: {classes}")
    print(f"Amostra de teste: classe real = {classes[sample_label]} | prevista = {classes[predicted_idx]}")
    print(f"Threads CPU: {args.threads}")
    print(f"Warm-up: {args.warmup} | Repetições medidas: {args.repeats}")
    print(f"Tamanho do modelo: {model_size_mb:.2f} MB")
    print(f"Latência média: {latencies_ms.mean():.4f} ms")
    print(f"Latência mediana: {np.median(latencies_ms):.4f} ms")
    print(f"Latência p95: {np.percentile(latencies_ms, 95):.4f} ms")
    print(f"Desvio padrão: {latencies_ms.std(ddof=1):.4f} ms")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()