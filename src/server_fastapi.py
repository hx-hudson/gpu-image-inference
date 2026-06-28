from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

import tensorrt as trt
import torch
from torchvision import transforms
from src.model import load_model
from src.benchmark_tensorrt import load_engine, inspect_engine

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

app = FastAPI(
    title="GPU Image Inference Server",
    description="FastAPI server for CIFAR-10 image classification",
    version="0.1.0"
)

device = torch.device("cuda")
model = load_model(
        r"D:\Py Project\gpu-image-inference\models\resnet_cifar10.pth",
        device
    )
model.to(device)
model.eval()

engine = load_engine(r"D:\Py Project\gpu-image-inference\models\resnet_cifar10_fp32.trt")
context = engine.create_execution_context()

input_name, output_name = inspect_engine(engine)
dtype = engine.get_tensor_dtype(input_name)

if dtype == trt.DataType.FLOAT:
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float16

def predict_pytorch(image):

    with torch.no_grad():
        logits = model(image)
        prediction = logits.argmax(dim=1).item()

    return prediction

def predict_tensorrt(image):
    input_shape = image.shape
    output_shape = (image.shape[0], 10)
    context.set_input_shape(input_name, input_shape)

    input_tensor = torch.tensor(
        image, device=device, dtype=torch_dtype
    ).contiguous()
    output_tensor = torch.empty(
        output_shape, device=device, dtype=torch_dtype
    ).contiguous()
    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())

    trt_stream = torch.cuda.Stream()
    trt_stream.wait_stream(torch.cuda.current_stream())
    context.execute_async_v3(trt_stream.cuda_stream)
    trt_stream.synchronize()

    prediction = output_tensor.argmax(dim=1).item()

    return prediction

@app.get("/")
def root():
    return {
        "message": "GPU Image Inference Server is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        backend: str = "pytorch"
):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    if backend == "pytorch":
        return {
            "class": CLASS_NAMES[predict_pytorch(image)]
        }
    elif backend == "tensorrt":
        return {
            "class": CLASS_NAMES[predict_tensorrt(image)]
        }

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported backend: {backend}"
    )