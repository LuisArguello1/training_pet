import torch
from torchvision import datasets, transforms
import os

DATA_DIR = "dog_weight_dataset"  # ruta a tu carpeta con train/ y val/

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)

print("Clases detectadas:", train_dataset.classes)
