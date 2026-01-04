import os
import shutil
import random
from pathlib import Path

# --- CONFIGURACIÓN ---
# Carpeta donde se descargó Kaggle (origen sucio)
RAW_DATA_DIR = Path("data/data1a") 
# Carpeta donde pondremos el dataset limpio
OUTPUT_DIR = Path("data/processed")

SPLIT_RATIOS = (0.6, 0.2, 0.2) # Train, Val, Test
SEED = 42 # Para reproducibilidad

def setup_directories():
    if OUTPUT_DIR.exists():
        print(f"⚠️ Borrando versión anterior en {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val', 'test']:
        for label in ['damage', 'whole']:
            (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

def get_all_files(class_name_contains):
    """Busca recursivamente todas las imágenes que coincidan con una etiqueta"""
    # Kaggle usa nombres como '00-damage' o '01-whole'
    files = []
    # Buscamos en todas las subcarpetas del raw dir
    for filepath in RAW_DATA_DIR.rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            # Check simple: si el nombre de la carpeta padre contiene la etiqueta
            if class_name_contains in filepath.parent.name:
                files.append(filepath)
    return files

def split_and_copy(files, label_dest):
    # 1. Mezclar
    random.shuffle(files)
    
    # 2. Calcular índices
    n = len(files)
    train_end = int(n * SPLIT_RATIOS[0])
    val_end = train_end + int(n * SPLIT_RATIOS[1])
    
    # 3. Trocear
    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }
    
    # 4. Copiar
    print(f"   procesando '{label_dest}': Total {n} -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    for split_name, split_files in splits.items():
        for file in split_files:
            dest = OUTPUT_DIR / split_name / label_dest / file.name
            shutil.copy2(file, dest)

def main():
    random.seed(SEED)
    
    if not RAW_DATA_DIR.exists():
        print(f"❌ No encuentro {RAW_DATA_DIR}. Asegúrate de haber ejecutado la descarga de Kaggle primero.")
        return

    print("🚀 Reorganizando dataset en 60/20/20...")
    setup_directories()
    
    # 1. Procesar Dañados (Buscamos carpetas que contengan 'damage')
    damage_files = get_all_files("damage")
    split_and_copy(damage_files, "damage")
    
    # 2. Procesar Intactos (Buscamos carpetas que contengan 'whole')
    whole_files = get_all_files("whole")
    split_and_copy(whole_files, "whole")
    
    print(f"\n✅ ¡Hecho! Tus datos limpios están en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()