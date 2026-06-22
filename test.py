import torch
import onnx
import onnxruntime as ort
import tensorrt as trt

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(onnx.__version__)
print(ort.__version__)
print(ort.get_available_providers())
print(trt.__version__)