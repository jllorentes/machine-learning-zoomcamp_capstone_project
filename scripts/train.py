import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import copy
import time

# --- 1. CONFIGURACIÓN (HIPERPARÁMETROS GANADORES) ---
# Configuración "Balanced"
LR = 0.001
EPOCHS = 15
SIZE_INNER = 128
DROPRATE = 0.3

# Configuración de Hardware/Datos
BATCH_SIZE = 16  # Bajo para no saturar la CPU local
IMG_SIZE = 224
DATA_DIR = "data/processed" # Asegúrate de haber ejecutado prepare_dataset.py
MODEL_SAVE_PATH = "car_damage.onnx"

# Detección de dispositivo (Soporte para Mac M1/M2 o CUDA si hubiera)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"🖥️ Usando dispositivo para entrenamiento: {device}")

# --- 2. DEFINICIÓN DEL MODELO ---
class CarDamageModel(nn.Module):
    def __init__(self, size_inner, droprate, num_classes=2):
        super(CarDamageModel, self).__init__()
        
        # Cargar Backbone MobileNetV3 Small
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        
        # Congelar pesos base
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Nueva "cabeza" personalizada
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

# --- 3. PREPARACIÓN DE DATOS ---
def get_dataloaders():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ No se encuentra {DATA_DIR}. Ejecuta 'uv run prepare_dataset.py' primero.")

    # Transformaciones idénticas a las usadas en el EDA/Colab
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
    # num_workers=0 es más seguro en Windows/Mac local para evitar errores de multiproceso
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_ds.classes

# --- 4. BUCLE DE ENTRENAMIENTO ---
def train_model():
    print("⏳ Cargando datos...")
    train_loader, val_loader, class_names = get_dataloaders()
    print(f"✅ Clases detectadas: {class_names}")

    # Instanciar Modelo
    model = CarDamageModel(size_inner=SIZE_INNER, droprate=DROPRATE, num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.backbone.classifier.parameters(), lr=LR)

    print(f"🚀 Iniciando entrenamiento por {EPOCHS} épocas...")
    start_time = time.time()
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        # Cada época tiene fase de entrenamiento y validación
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            
            # Iterar sobre datos (con print simple de progreso)
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
                
                # Feedback visual cada 10 batches para no desesperar en CPU
                if phase == 'train' and i % 10 == 0:
                    print(f"\r   Batch {i}/{len(dataloader)} procesado...", end="")

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'\n   {phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Guardar si es el mejor
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print(f'\n🏁 Entrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🏆 Mejor Val Acc: {best_acc:.4f}')

    # Cargar mejores pesos
    model.load_state_dict(best_model_wts)
    return model

# --- 5. EXPORTAR ---
def export_onnx(model):
    print(f"💾 Exportando modelo a {MODEL_SAVE_PATH}...")
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
    print("✅ ¡Modelo exportado y listo para producción!")

if __name__ == "__main__":
    try:
        final_model = train_model()
        export_onnx(final_model)
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento cancelado por el usuario.")
    except Exception as e:
        print(f"\n❌ Error: {e}")