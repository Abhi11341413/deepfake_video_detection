import os
import torch
import numpy as np
import argparse
import glob

from src.models.deepfake_detector import JointDetector
from src.preprocessing.mouth_crop import MouthCropper
from src.preprocessing.audio_extract import AudioExtractor

def run_demo(video_path):
    print("\n" + "="*50)
    print(f" DEEPFAKE DETECTOR DEMO ")
    print("="*50)
    print(f"\nAnalyzing Video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return

    print("\n[1/3] Extracting Features...")
    # Initialize preprocessors
    vis_cropper = MouthCropper()
    aud_extractor = AudioExtractor()
    
    # Process visual and audio
    print("  -> Tracking facial landmarks and cropping mouth...")
    vis_success = vis_cropper.process_video(video_path, output_filename="demo_mouth.npy")
    print("  -> Extracting audio frequencies (MFCCs)...")
    aud_success = aud_extractor.extract_mfcc(video_path, output_filename="demo_audio.npy")
    
    if not (vis_success and aud_success):
        print("❌ Error: Failed to extract features from the video. Ensure the video has a face and audio track.")
        return
        
    print("\n[2/3] Loading AI Model...")
    # Find the latest checkpoint
    checkpoint_dir = "lightning_logs"
    if not os.path.exists(checkpoint_dir):
        print("❌ Error: No trained model found! Please train the model first.")
        return
        
    # Get the most recent checkpoint
    list_of_files = glob.glob(f'{checkpoint_dir}/**/*.ckpt', recursive=True)
    if not list_of_files:
        print("❌ Error: No trained model checkpoints found!")
        return
        
    latest_checkpoint = max(list_of_files, key=os.path.getctime)
    
    # Load model
    model = JointDetector.load_from_checkpoint(latest_checkpoint)
    model.eval() # Set to evaluation mode
    
    print("\n[3/3] Running Inference...")
    # Load the processed arrays
    vis_data = np.load("data/processed/demo_mouth.npy")
    aud_data = np.load("data/processed/demo_audio.npy")
    
    # Convert to PyTorch tensors and add Batch dimension [1, ...]
    vis_tensor = torch.tensor(vis_data).float().unsqueeze(0)
    aud_tensor = torch.tensor(aud_data).float().unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        logits = model(vis_tensor, aud_tensor)
        probability = torch.sigmoid(logits).item()
        
    print("\n" + "="*50)
    print(" PREDICTION RESULTS")
    print("="*50)
    
    if probability >= 0.5:
        confidence = probability * 100
        print(f" RESULT: FAKE VIDEO DETECTED ")
    else:
        confidence = (1 - probability) * 100
        print(f" RESULT: REAL VIDEO (AUTHENTIC) ")
        
    print(f"Confidence Score: {confidence:.2f}%")
    print("="*50 + "\n")
    
    # Clean up demo files
    os.remove("data/processed/demo_mouth.npy")
    os.remove("data/processed/demo_audio.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detector Demo")
    parser.add_argument("--video", type=str, help="Path to the video file to analyze", required=True)
    args = parser.parse_args()
    
    run_demo(args.video)
