import torch
import torch.nn as nn
from torchvision import models

def build_model():

    model = models.resnet18()
    model.conv1 = nn.Conv2d(3, 64, 3,
                            stride=1, padding=1)
    model.maxpool = nn.Identity()

    in_feature = model.fc.in_features
    model.fc = nn.Linear(in_feature, 10)

    return model

def load_model(
        checkpoint_path: str,
        device: torch.device
) -> nn.Module:

    model = build_model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model