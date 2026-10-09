import os
import glob
from src.preprocessing.mouth_crop import MouthCropper
from src.preprocessing.audio_extract import AudioExtractor

def batch_process(raw_dir="data/raw", processed_dir="data/processed"):
    print(f"Starting batch processing from {raw_dir}...")
    
    # Initialize our processors
    vis_cropper = MouthCropper()
    aud_extractor = AudioExtractor()
    
    
    video_paths = glob.glob(os.path.join(raw_dir, '**', '*.mp4'), recursive=True)
    
    if not video_paths:
        print(f"No .mp4 files found in {raw_dir}. Please make sure you have downloaded the videos.")
        return
        
    print(f"Found {len(video_paths)} videos to process.")
    
    success_count = 0
    for idx, video_path in enumerate(video_paths):
        print(f"\n[{idx+1}/{len(video_paths)}] Processing {video_path}...")
        
        parts = video_path.split(os.sep)
        label_prefix = ""
        if "real" in parts:
            label_prefix = "real_"
        elif "fake" in parts:
            label_prefix = "fake_"
            
        base_name = os.path.basename(video_path).split('.')[0]
        
        # output filenames
        vis_filename = f"{label_prefix}{base_name}_mouth.npy"
        aud_filename = f"{label_prefix}{base_name}_audio.npy"
        
        # Process Video Stream
        print("  -> Extracting mouth sequence...")
        vis_success = vis_cropper.process_video(video_path, output_filename=vis_filename)
        
        # Process Audio Stream
        print("  -> Extracting audio MFCCs...")
        aud_success = aud_extractor.extract_mfcc(video_path, output_filename=aud_filename)
        
        if vis_success and aud_success:
            print("  -> Successfully processed both streams!")
            success_count += 1
        else:
            print("  -> Failed to process one or both streams for this video.")
            
    print(f"\nBatch processing complete! Successfully processed {success_count}/{len(video_paths)} videos.")

if __name__ == "__main__":
    batch_process()
