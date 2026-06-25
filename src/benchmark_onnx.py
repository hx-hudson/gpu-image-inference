import os
import time
import onnx
import onnxruntime
import torch
import numpy as np
import argparse
from benchmark_pytorch import print_results, save_results

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--batch_size", type=int, nargs="+", default=[1,8,16,32,64])
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    return parser.parse_args()

def measure(
        ort_session,
        x,
        warmup: int,
        repeats: int
):
    # warm up
    for _ in range(warmup):
        ort_session.run(["output"], {"input": x})


    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        ort_session.run(["output"], {"input": x})

    torch.cuda.synchronize()
    end = time.perf_counter()

    time_cost = end - start
    latency_ms = time_cost / repeats * 1000
    throughput = x.shape[0] * repeats / time_cost

    return latency_ms, throughput

def main():

    args = get_args()

    # Get path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    if args.load_path is None:
        load_path = os.path.join(root_dir, "models", "resnet_cifar10.onnx")
    else:
        load_path = args.load_path
    if args.output_path is None:
        output_path = os.path.join(root_dir, "results", "benchmark_onnx_fp32.csv")
    else:
        output_path = args.output_path


    # Load model
    ort_session = onnxruntime.InferenceSession(
        load_path, providers=["CUDAExecutionProvider"]
    )

    # Measure
    rows = []
    for batch_size in args.batch_size:
        x = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)

        latency_ms, throughput = measure(ort_session, x, args.warmup, args.repeats)

        rows.append({
            "Backend": "ONNX",
            "Precision": "FP32",
            "Batch Size": batch_size,
            "Latency(ms/batch)": latency_ms,
            "Throughput(img/s)": throughput
        })

    print_results(rows)
    save_results(rows, output_path)

if __name__ == "__main__":
    main()