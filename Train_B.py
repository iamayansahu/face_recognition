import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models
from torchvision.transforms import GaussianBlur
from sklearn.metrics import classification_report
import os
import time
import multiprocessing
from collections import Counter

# Configuration
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 224
TRAIN_DIR = r"Task_B/train"
VAL_DIR = r"Task_B/val"
MODEL_PATH = 'Face_Recognition_Model.pth'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("✅ CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🚀 Using GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ Using CPU — training will be slower")

# Transforms
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomApply([GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform_val)
class_names = train_dataset.classes


class_counts = Counter(train_dataset.targets)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model (ResNet50)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.BatchNorm1d(model.fc.in_features),
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, len(class_names))
)
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(weight=None, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

# Training
best_acc = 0.0
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {acc:.4f} | Best: {best_acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("💾 Best model saved.")

# Final report
duration = (time.time() - start_time) / 60
print(f"\n⏱ Training complete in {duration:.2f} minutes")
print("\n📊 Final Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == '__main__':
    multiprocessing.freeze_support()