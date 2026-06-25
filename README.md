# GPU-Accelerated Image Classification Inference System

This project benchmarks image classification inference performance on GPU using a ResNet18 model trained for CIFAR-10.

Current benchmark focus:

- PyTorch FP32 baseline
- ONNX Runtime FP32 with `CUDAExecutionProvider`

The model input shape is `[B, 3, 32, 32]`, where `B` is the batch size.

---

## Environment

Benchmark device:

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- Dataset/model: CIFAR-10 ResNet18
- Precision: FP32
- Input shape: `[B, 3, 32, 32]`

Benchmark settings:

- Warmup iterations: 20
- Benchmark iterations: 500
- CUDA synchronization before and after timing
- Batch sizes: 1, 8, 16, 32, 64

---

## PyTorch FP32 Baseline

Command:

```bash
python src/benchmark_pytorch.py --repeats 500
```

Results saved to:

```text
results/benchmark_pytorch_fp32.csv
```

| Backend | Precision | Batch Size | Latency(ms/batch) | Throughput(img/s) |
|---|---|---:|---:|---:|
| PyTorch | FP32 | 1 | 1.479 | 676.23 |
| PyTorch | FP32 | 8 | 1.912 | 4184.04 |
| PyTorch | FP32 | 16 | 1.749 | 9149.17 |
| PyTorch | FP32 | 32 | 2.435 | 13142.89 |
| PyTorch | FP32 | 64 | 5.472 | 11696.24 |

---

## ONNX Runtime FP32 Benchmark

Command:

```bash
python src/benchmark_onnx.py --repeats 500
```

Execution provider:

```text
CUDAExecutionProvider
```

Results saved to:

```text
results/benchmark_onnx_fp32.csv
```

| Backend | Precision | Batch Size | Latency(ms/batch) | Throughput(img/s) |
|---|---|---:|---:|---:|
| ONNX Runtime | FP32 | 1 | 1.083 | 923.17 |
| ONNX Runtime | FP32 | 8 | 1.324 | 6040.54 |
| ONNX Runtime | FP32 | 16 | 1.414 | 11317.66 |
| ONNX Runtime | FP32 | 32 | 2.214 | 14455.63 |
| ONNX Runtime | FP32 | 64 | 5.397 | 11858.43 |

---

## PyTorch vs ONNX Runtime FP32

| Batch Size | PyTorch Latency(ms) | ONNX Runtime Latency(ms) | Latency Speedup | PyTorch Throughput(img/s) | ONNX Runtime Throughput(img/s) | Throughput Gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.479 | 1.083 | 1.37x | 676.23 | 923.17 | +36.52% |
| 8 | 1.912 | 1.324 | 1.44x | 4184.04 | 6040.54 | +44.37% |
| 16 | 1.749 | 1.414 | 1.24x | 9149.17 | 11317.66 | +23.70% |
| 32 | 2.435 | 2.214 | 1.10x | 13142.89 | 14455.63 | +9.99% |
| 64 | 5.472 | 5.397 | 1.01x | 11696.24 | 11858.43 | +1.39% |

---

## Observations

ONNX Runtime FP32 is faster than PyTorch FP32 for all tested batch sizes.

The improvement is most obvious at smaller and medium batch sizes. At batch size 1, ONNX Runtime reduces latency from 1.479 ms to 1.083 ms. At batch size 8, it reduces latency from 1.912 ms to 1.324 ms.

At larger batch sizes, the gap becomes smaller. Batch size 64 shows only a small difference between PyTorch and ONNX Runtime. This suggests that PyTorch is already using the GPU efficiently at larger batches, while ONNX Runtime provides a clearer advantage when launch overhead and framework overhead are more significant.

Current ONNX Runtime benchmarking uses regular `session.run()` with NumPy inputs. This means the measured latency may include CPU-to-GPU input transfer and GPU-to-CPU output transfer. A future optimization is to use ONNX Runtime I/O Binding so that input and output buffers stay on the GPU.

---
