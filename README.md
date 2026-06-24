## PyTorch FP32 Baseline Benchmark

The PyTorch FP32 inference benchmark was measured on an NVIDIA GeForce RTX 5080 Laptop GPU using ResNet18 on CIFAR-10 input shape `[B, 3, 32, 32]`.

| Backend | Precision | Batch Size | Latency(ms/batch) | Throughput(img/s) |
|---|---|---:|---:|---:|
| PyTorch | FP32 | 1 | 1.479 | 676.23 |
| PyTorch | FP32 | 8 | 1.912  | 4184.04 |
| PyTorch | FP32 | 16 | 1.749 | 9149.17 |
| PyTorch | FP32 | 32 | 2.435 | 13142.89 |
| PyTorch | FP32 | 64 | 5.472 | 11696.24 |

Benchmark settings:
- warmup iterations: 20
- benchmark iterations: 500
- CUDA synchronization before and after timing
- `torch.inference_mode()` enabled
- FP32 inference