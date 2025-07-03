import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ====== Configuration ======
MODEL_PATH = r"Face_Recognition_Model.pth"  # Your saved model
TRAIN_DIR = r"Task_B_Dataset/train"  # For class names
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load Class Names ======
class_names = sorted(os.listdir(TRAIN_DIR))

# ====== Define Transforms ======
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


# ====== Load Model ======
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(model.fc.in_features),
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(class_names))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


# ====== Predict Function ======
def predict(image_path):
    model = load_model()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_class = class_names[pred.item()]
        confidence = torch.softmax(outputs, dim=1)[0][pred.item()].item()

    print(f"📸 Image: {os.path.basename(image_path)}")
    print(f"🧠 Predicted Class: {predicted_class}")
    print(f"✅ Confidence: {confidence * 100:.2f}%")


# ====== Example Usage ======
if __name__ == "__main__":
    test_image_path = r"Aishwarya_Rai.jpg"  # ⬅ Replace with your test image path
    predict(test_image_path)