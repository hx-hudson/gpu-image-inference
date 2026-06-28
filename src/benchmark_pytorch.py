import os
import csv
import time
import torch
import argparse
from src.model import load_model
import torch.nn as nn

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--batch_size", type=int, nargs="+", default=[1,8,16,32,64])

    return parser.parse_args()

def measure(
        model: nn.Module,
        x: torch.Tensor,
        warmup: int,
        repeat: int
):
    # warm up
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(repeat):
            model(x)

    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    latency_ms = total_time / repeat *1000
    throughput = x.shape[0] * repeat / total_time

    return latency_ms, throughput

def print_results(rows):
    headers = [
        "Backend",
        "Precision",
        "Batch Size",
        "Latency(ms/batch)",
        "Throughput(img/s)"
    ]

    table = []

    for row in rows:
        table.append([
            row["Backend"],
            row["Precision"],
            str(row["Batch Size"]),
            f"{row['Latency(ms/batch)']:.3f}",
            f"{row['Throughput(img/s)']:.2f}"
        ])

    col_widths = []

    for col_idx in range(len(headers)):
        max_width = len(headers[col_idx])

        for row in table:
            max_width = max(max_width, len(row[col_idx]))

        col_widths.append(max_width)

    header_line = " | ".join(
        headers[i].ljust(col_widths[i])
        for i in range(len(headers))
    )

    separator_line = "-+-".join(
        "-" * col_widths[i]
        for i in range(len(headers))
    )

    print()
    print(header_line)
    print(separator_line)

    # 打印每一行
    for row in table:
        line = " | ".join(
            row[i].ljust(col_widths[i])
            for i in range(len(row))
        )
        print(line)

def save_results(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "Backend",
        "Precision",
        "Batch Size",
        "Latency(ms/batch)",
        "Throughput(img/s)"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    args = get_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)

    if args.checkpoint is None:
        checkpoint_path = os.path.join(root_dir, "models", "resnet_cifar10.pth")
    else:
        checkpoint_path = args.checkpoint

    if args.output is None:
        output_path = os.path.join(root_dir, "results", "benchmark_pytorch_fp32.csv")
    else:
        output_path = args.output

    # Strict FP32 baseline
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda")
    model = load_model(checkpoint_path, device)
    model.eval()

    rows = []

    for batch_size in args.batch_size:

        x = torch.randn((batch_size,3,32,32), device=device, dtype=torch.float32)
        latency_ms, throughput = measure(model, x, args.warmup, args.repeats)

        rows.append({
            "Backend": "PyTorch",
            "Precision": "FP32",
            "Batch Size": batch_size,
            "Latency(ms/batch)": latency_ms,
            "Throughput(img/s)": throughput
        })

    print_results(rows)
    save_results(rows, output_path)

if __name__ == "__main__":
    main()
