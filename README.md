# GPU-Accelerated Image Classification Inference System

This project benchmarks GPU inference performance for a ResNet18 CIFAR-10 image classification model.

Tested backends:

- PyTorch FP32
- ONNX Runtime FP32 (`CUDAExecutionProvider`)
- TensorRT FP32
- TensorRT FP16

Input shape: `[B, 3, 32, 32]`

---

## Environment

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- Model: ResNet18 for CIFAR-10
- Batch sizes: 1, 8, 16, 32, 64
- Warmup iterations: 20
- Benchmark iterations: 500

---

## Run Benchmarks

```bash
python src/benchmark_pytorch.py --repeats 500
python src/benchmark_onnx.py --repeats 500
python src/benchmark_tensorrt.py --load_path models/resnet_cifar10_fp32.trt --repeats 500
python src/benchmark_tensorrt.py --load_path models/resnet_cifar10_fp16.trt --repeats 500
```

Results are saved under:

```text
results/
```

---

## Benchmark Results

| Backend | Precision | Batch Size | Latency(ms/batch) | Throughput(img/s) |
|---|---|---:|---:|---:|
| PyTorch | FP32 | 1 | 1.479 | 676.23 |
| PyTorch | FP32 | 8 | 1.912 | 4184.04 |
| PyTorch | FP32 | 16 | 1.749 | 9149.17 |
| PyTorch | FP32 | 32 | 2.435 | 13142.89 |
| PyTorch | FP32 | 64 | 5.472 | 11696.24 |
| ONNX Runtime | FP32 | 1 | 1.083 | 923.17 |
| ONNX Runtime | FP32 | 8 | 1.324 | 6040.54 |
| ONNX Runtime | FP32 | 16 | 1.414 | 11317.66 |
| ONNX Runtime | FP32 | 32 | 2.214 | 14455.63 |
| ONNX Runtime | FP32 | 64 | 5.397 | 11858.43 |
| TensorRT | FP32 | 1 | 0.753 | 1328.63 |
| TensorRT | FP32 | 8 | 0.894 | 8945.17 |
| TensorRT | FP32 | 16 | 1.014 | 15772.58 |
| TensorRT | FP32 | 32 | 1.700 | 18829.04 |
| TensorRT | FP32 | 64 | 2.965 | 21586.65 |
| TensorRT | FP16 | 1 | 0.335 | 2984.57 |
| TensorRT | FP16 | 8 | 0.353 | 22634.89 |
| TensorRT | FP16 | 16 | 0.436 | 36734.79 |
| TensorRT | FP16 | 32 | 0.563 | 56801.66 |
| TensorRT | FP16 | 64 | 0.961 | 66591.76 |

---

## Summary

TensorRT gives the best inference performance in this benchmark.

At batch size 64:

- PyTorch FP32: 11,696.24 img/s
- ONNX Runtime FP32: 11,858.43 img/s
- TensorRT FP32: 21,586.65 img/s
- TensorRT FP16: 66,591.76 img/s

TensorRT FP16 reaches about **5.69x higher throughput** than the PyTorch FP32 baseline at batch size 64.

---

## Notes

The ONNX Runtime benchmark uses regular `session.run()` with NumPy inputs, so the measured time may include CPU-GPU data transfer overhead. TensorRT benchmark uses GPU tensors directly as input and output buffers.
