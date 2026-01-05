import os
import shutil
import random
from pathlib import Path

# --- SETINGS ---

RAW_DATA_DIR = Path("data/raw_temp") 
OUTPUT_DIR = Path("data/processed")

SPLIT_RATIOS = (0.6, 0.2, 0.2) # Train, Val, Test
SEED = 42 # For reproducibility
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(".")
os.environ['KAGGLE_CONFIG_DIR'] = str(PROJECT_ROOT)
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset():
    
    
    if RAW_DATA_DIR.exists():
        print("The temporary download folder already exists. Skipping download..")
        return

    print("Downloading dataset from Kaggle...")
    api = KaggleApi()
    api.authenticate()
    
    api.dataset_download_files("anujms/car-damage-detection", path=RAW_DATA_DIR, unzip=True)
    
    for zip_file in RAW_DATA_DIR.glob("*.zip"):
        zip_file.unlink()
        
    print(f"Download completed in: {RAW_DATA_DIR}")

def setup_directories():
    if OUTPUT_DIR.exists():
        print(f"Deleting previous version in{OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val', 'test']:
        for label in ['damage', 'whole']:
            (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

def get_all_files(class_name_contains):
    """Recursively search for all images that match a tag."""
    files = []
    for filepath in RAW_DATA_DIR.rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            if class_name_contains in filepath.parent.name:
                files.append(filepath)
    return files

def split_and_copy(files, label_dest):
    random.shuffle(files)
    
    n = len(files)
    train_end = int(n * SPLIT_RATIOS[0])
    val_end = train_end + int(n * SPLIT_RATIOS[1])
    
    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    print(f"   processing '{label_dest}': Total {n} -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    for split_name, split_files in splits.items():
        for file in split_files:
            dest = OUTPUT_DIR / split_name / label_dest / file.name
            shutil.copy2(file, dest)

def main():
    
    download_dataset()

    random.seed(SEED)
    
    if not RAW_DATA_DIR.exists():
        print(f"Cannot find {RAW_DATA_DIR}. Please ensure that you have downloaded the Kaggle data first.")
        return

    print("Reorganising dataset into 60/20/20...")
    setup_directories()
    
    damage_files = get_all_files("damage")
    split_and_copy(damage_files, "damage")
    
    whole_files = get_all_files("whole")
    split_and_copy(whole_files, "whole")
    
    print(f"\n Done! Clean data is now in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()