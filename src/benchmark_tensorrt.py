import os
import argparse
import torch
import tensorrt as trt
import time
from src.benchmark_pytorch import print_results, save_results

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--batch_size", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    parser.add_argument("--output_path", type=str, default=None)

    return parser.parse_args()

def load_engine(load_path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(load_path, "rb") as f:
        engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    return engine

def inspect_engine(engine):
    # get input and output name
    input_name = ''
    output_name = ''
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        if mode == trt.TensorIOMode.OUTPUT:
            output_name = name

    return input_name, output_name

def measure(context, batch_size, warmup, repeats):
    # create stream
    trt_stream = torch.cuda.Stream()
    trt_stream.wait_stream(torch.cuda.current_stream())

    for _ in range(warmup):
        context.execute_async_v3(trt_stream.cuda_stream)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        context.execute_async_v3(trt_stream.cuda_stream)

    torch.cuda.synchronize()
    end = time.perf_counter()

    time_cost = end - start
    latency_ms = time_cost / repeats * 1000
    throughput = batch_size * repeats / time_cost

    return latency_ms, throughput


def main():
    args = get_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)

    if args.load_path is None:
        load_path = os.path.join(
            root_dir, "models", "resnet_cifar10_fp32.trt"
        )
    else:
        load_path = args.load_path



    engine = load_engine(load_path)

    # create context
    context = engine.create_execution_context()

    input_name, output_name = inspect_engine(engine)

    dtype = engine.get_tensor_dtype(input_name)
    if args.output_path is None:
        if dtype == trt.DataType.FLOAT:
            output_path = os.path.join(
                root_dir, "results", "benchmark_tensorrt_fp32.csv"
            )
        else:
            output_path = os.path.join(
                root_dir, "results", "benchmark_tensorrt_fp16.csv"
            )
    else:
        output_path = args.output_path

    if dtype == trt.DataType.FLOAT:
        torch_dtype = torch.float32
        precision = "FP32"
    else:
        torch_dtype = torch.float16
        precision = "FP16"

    rows = []
    for batch_size in args.batch_size:

        input_shape = (batch_size, 3, 32, 32)
        output_shape = (batch_size, 10)
        device = torch.device("cuda")

        context.set_input_shape(input_name, input_shape)

        input_tensor = torch.randn(
            input_shape, device=device, dtype=torch_dtype
        ).contiguous()
        output_tensor = torch.empty(
            output_shape, device=device, dtype=torch_dtype
        ).contiguous()
        context.set_tensor_address(input_name, input_tensor.data_ptr())
        context.set_tensor_address(output_name, output_tensor.data_ptr())

        latency_ms, throughput = measure(
            context, batch_size, args.warmup, args.repeats
        )

        rows.append({
            "Backend": "TensorRT",
            "Precision": precision,
            "Batch Size": batch_size,
            "Latency(ms/batch)": latency_ms,
            "Throughput(img/s)": throughput
        })

    print_results(rows)
    save_results(rows, output_path)

if __name__ == "__main__":
    main()