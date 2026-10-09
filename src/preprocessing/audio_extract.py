import os
import xml.etree.ElementTree as ET
import tempfile
import cv2
import librosa
import numpy as np


from moviepy import VideoFileClip

class AudioExtractor:
    def __init__(self, config_path="config.xml"):
        self.config_path = config_path
        self._load_config()
        self.n_mfcc = 40
        
    def _load_config(self):
        """Loads hyperparameter configuration from config.xml."""
        try:
            tree = ET.parse(self.config_path)
            root = tree.getroot()
            self.frames_per_sequence = int(root.find(".//FramesPerSequence").text)
            self.output_dir = root.find(".//ProcessedData").text
        except Exception as e:
            print(f"Error reading config: {e}")
            self.frames_per_sequence = 20
            self.output_dir = "data/processed"
            
        os.makedirs(self.output_dir, exist_ok=True)
        
    def extract_mfcc(self, video_path, output_filename=None):
        """
        Extracts audio from video, computes 40 MFCCs, and resizes to match 
        the temporal sequence length of the visual stream.
        Output shape: [Channels (3), MFCC (40), Time (20)]
        """
        base_name = os.path.basename(video_path).split('.')[0]
        if output_filename is None:
            output_filename = f"{base_name}_audio.npy"
            
        try:
            
            video = VideoFileClip(video_path)
            if video.audio is None:
                print(f"No audio track found in {video_path}")
                video.close()
                return False
                
            temp_wav = tempfile.mktemp(suffix='.wav')
           
            video.audio.write_audiofile(temp_wav, logger=None)
            video.close()
            
            y, sr = librosa.load(temp_wav, sr=16000)
            os.remove(temp_wav)
            
          
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
           
            mfcc_resized = cv2.resize(mfcc, (self.frames_per_sequence, self.n_mfcc), interpolation=cv2.INTER_LINEAR)
          
            mfcc_tensor = np.expand_dims(mfcc_resized, axis=0)
            # Shape: [3, 40, 20]
            mfcc_tensor = np.repeat(mfcc_tensor, 3, axis=0)
            
            mfcc_tensor = (mfcc_tensor - np.mean(mfcc_tensor)) / (np.std(mfcc_tensor) + 1e-6)
            
            save_path = os.path.join(self.output_dir, output_filename)
            np.save(save_path, mfcc_tensor)
            print(f"Saved audio sequence to {save_path} with shape {mfcc_tensor.shape}")
            return True
            
        except Exception as e:
            print(f"Error processing audio from {video_path}: {e}")
            return False

if __name__ == "__main__":
    print("AudioExtractor initialized. Please instantiate and call extract_mfcc(video_path).")
