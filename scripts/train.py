import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import copy
import time
import sys
from pathlib import Path

# --- 1. SETTINGS ---
# Balanced configuration from Colab experiments
LR = 0.01
EPOCHS = 15
SIZE_INNER = 128
DROPRATE = 0.3

BATCH_SIZE = 16
IMG_SIZE = 224

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_SAVE_PATH = BASE_DIR / "best_car_damage_model.onnx"


# Device detection (Support for Mac M1/M2 or CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using a training device: {device}")

# --- 2. MODEL---
class CarDamageModel(nn.Module):
    def __init__(self, size_inner, droprate, num_classes=2):
        super(CarDamageModel, self).__init__()
        
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        input_features = 576
        self.classifier = nn.Sequential(
            nn.Linear(input_features, size_inner),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(size_inner, num_classes)
        )
        self.backbone.classifier = self.classifier

    def forward(self, x):
        return self.backbone(x)

# --- 3. DATA PREPARATION ---
def get_dataloaders():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"{DATA_DIR} cannot be found. Run 'uv run prepare_dataset.py' first.")

    transforms_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transforms_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transforms_train)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transforms_val)

    # Dataloaders
    # num_workers=0 safer on local Mac
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_ds.classes

# --- 4. TRAINING---
def train_model():
    print("Loading data...")
    train_loader, val_loader, class_names = get_dataloaders()
    print(f"Classes detected: {class_names}")

    model = CarDamageModel(size_inner=SIZE_INNER, droprate=DROPRATE, num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.backbone.classifier.parameters(), lr=LR)

    print(f"Starting training for {EPOCHS} epochs...")
    start_time = time.time()
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train' and i % 10 == 0:
                    print(f"\r   Batch {i}/{len(dataloader)} processed...", end="")

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'\n   {phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print(f'\n🏁 Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🏆 Best Val Acc: {best_acc:.4f}')

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model

# --- 5. EXPORT ---
def export_onnx(model):
    print(f"💾 Exporting model to {MODEL_SAVE_PATH}...")
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_SAVE_PATH,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Model exported and ready for production!")

if __name__ == "__main__":
    try:
        final_model = train_model()
        export_onnx(final_model)
    except KeyboardInterrupt:
        print("\n Training cancelled by the user.")
    except Exception as e:
        print(f"\n Error: {e}")