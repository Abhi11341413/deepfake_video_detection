import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import xml.etree.ElementTree as ET

from src.models.deepfake_detector import JointDetector

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, length=100):
       
        self.data_dir = data_dir
        self.length = length
        self.has_real_data = False
        
       
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            mouth_files = [f for f in files if f.endswith('_mouth.npy')]
            if mouth_files:
                self.has_real_data = True
                self.mouth_files = mouth_files
                self.length = len(mouth_files)
                print(f"Found {self.length} processed samples in {data_dir}.")
            else:
                print("No processed data found. Using dummy data for training pipeline verification.")
        else:
            print("Data directory not found. Using dummy data for training pipeline verification.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.has_real_data:
            import numpy as np
            mouth_file = self.mouth_files[idx]
            base_name = mouth_file.replace('_mouth.npy', '')
            audio_file = base_name + '_audio.npy'
            
            mouth_path = os.path.join(self.data_dir, mouth_file)
            audio_path = os.path.join(self.data_dir, audio_file)
            
            try:
                visual_data = np.load(mouth_path)
                audio_data = np.load(audio_path)
            except Exception as e:
                print(f"Error loading {mouth_path} or {audio_path}: {e}")
                visual_data = np.zeros((20, 3, 112, 112), dtype=np.float32)
                audio_data = np.zeros((3, 40, 20), dtype=np.float32)
            
            # Extract label from filename (fake = 1, real = 0)
            label = 1 if mouth_file.startswith('fake') else 0
            
            return torch.tensor(visual_data).float(), torch.tensor(audio_data).float(), label
            
        visual_data = torch.randn(20, 3, 112, 112)
        # Audio: [Channels=3, MFCC=40, Time=20]
        audio_data = torch.randn(3, 40, 20)
        # Label: 0 or 1
        label = torch.randint(0, 2, (1,)).item()
        
        return visual_data, audio_data, label

def get_config():
    try:
        tree = ET.parse("config.xml")
        root = tree.getroot()
        batch_size = int(root.find(".//BatchSize").text)
        lr = float(root.find(".//LearningRate").text)
        data_dir = root.find(".//ProcessedData").text
        return batch_size, lr, data_dir
    except Exception as e:
        print(f"Failed to read config: {e}. Using defaults.")
        return 8, 1e-3, "data/processed"

def main():
    batch_size, lr, data_dir = get_config()
    
    # 1. Dataset & DataLoader
    dataset = DeepfakeDataset(data_dir=data_dir, length=32) # Small dataset for fast test
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model
    print("Initializing JointDetector...")
    model = JointDetector(learning_rate=lr)
    
    # 3. Trainer
    print("Starting local training loop...")
    trainer = pl.Trainer(max_epochs=15, log_every_n_steps=1)
    
    # 4. Train
    trainer.fit(model, dataloader)
    print("Training loop completed successfully!")

if __name__ == "__main__":
    main()
