from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from PIL import Image

import torch
from torchvision import transforms
from model import load_model
from

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
model = load_model()
model.to(device)
model.eval()

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
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        prediction = logits.argmax(dim=1)

    return {
        "class": CLASS_NAMES[prediction]
    }