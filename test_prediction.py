import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# =====================
# CONFIGURACIÓN
# =====================
MODEL_PATH = "models/dog_age_classifier.pth"
IMAGE_PATH = "img/senior.webp"  # Cambia esto por la imagen que quieras probar
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['adulto', 'cachorro', 'joven', 'senior']

# =====================
# CARGAR MODELO
# =====================
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model = model.to(DEVICE)

# =====================
# TRANSFORMACIÓN DE IMAGEN
# =====================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# CARGAR IMAGEN
# =====================
# Opción 1: con PIL
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # añadir batch dimension

# =====================
# PREDICCIÓN
# =====================
with torch.no_grad():
    output = model(input_tensor)
    _, pred = torch.max(output, 1)
    predicted_class = CLASS_NAMES[pred.item()]

print(f"La imagen se clasifica como: {predicted_class}")
