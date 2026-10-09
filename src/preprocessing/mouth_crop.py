import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import xml.etree.ElementTree as ET

class MouthCropper:
    def __init__(self, config_path="config.xml"):
        self.config_path = config_path
        self._load_config()
        
        # Initialize MediaPipe Face Landmarker
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Approximate indices for the mouth region
        self.MOUTH_INDICES = list(range(0, 15)) + list(range(17, 18)) + [37, 39, 40, 61, 267, 269, 270, 291, 314, 405]

    def _load_config(self):
        """Loads hyperparameter configuration from config.xml."""
        try:
            tree = ET.parse(self.config_path)
            root = tree.getroot()
            self.frames_per_sequence = int(root.find(".//FramesPerSequence").text)
            self.mouth_width = int(root.find(".//MouthCropSize/Width").text)
            self.mouth_height = int(root.find(".//MouthCropSize/Height").text)
            self.output_dir = root.find(".//ProcessedData").text
        except Exception as e:
            print(f"Error reading config: {e}")
            self.frames_per_sequence = 20
            self.mouth_width = 112
            self.mouth_height = 112
            self.output_dir = "data/processed"
            
        os.makedirs(self.output_dir, exist_ok=True)

    def crop_mouth(self, image, landmarks):
        """Crops the mouth region based on MediaPipe landmarks."""
        h, w, _ = image.shape
        
        # Gather mouth points
        mouth_points = []
        for idx in self.MOUTH_INDICES:
            # Depending on mediapipe version and exact indices, handle out-of-bounds gracefully
            if idx < len(landmarks):
                pt = landmarks[idx]
                mouth_points.append((int(pt.x * w), int(pt.y * h)))
        
        if not mouth_points:
            return None
            
        x_coords = [p[0] for p in mouth_points]
        y_coords = [p[1] for p in mouth_points]
        
        # Find bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        pad_x = int((x_max - x_min) * 0.2)
        pad_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        
        # Crop
        mouth_crop = image[y_min:y_max, x_min:x_max]
        if mouth_crop.size == 0:
            return None
            
        # Resize to standard shape (W, H)
        mouth_crop_resized = cv2.resize(mouth_crop, (self.mouth_width, self.mouth_height))
        
        # Convert to RGB (OpenCV uses BGR)
        mouth_crop_rgb = cv2.cvtColor(mouth_crop_resized, cv2.COLOR_BGR2RGB)
        
        # Shape: [Channels, Height, Width]
        mouth_crop_tensor = np.transpose(mouth_crop_rgb, (2, 0, 1))
        
        # Normalize to [0, 1]
        mouth_crop_tensor = mouth_crop_tensor.astype(np.float32) / 255.0
        return mouth_crop_tensor

    def process_video(self, video_path, output_filename=None):
        """
        Extracts sequence of mouth frames from video.
        Outputs a numpy array of shape [Batch, Channels, Time, Height, Width] (Batch=1)
        or [Time, Channels, Height, Width] and saves it as .npy.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video stream or file: {video_path}")
            return False
            
        frame_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert the BGR image to RGB before processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            
            if results.face_landmarks:
                # We assume max_num_faces=1
                landmarks = results.face_landmarks[0]
                mouth_crop = self.crop_mouth(frame, landmarks)
                
                if mouth_crop is not None:
                    frame_sequence.append(mouth_crop)
                    
            if len(frame_sequence) == self.frames_per_sequence:
                break
                
        cap.release()
        
        if len(frame_sequence) < self.frames_per_sequence:
            print(f"Warning: Extracted {len(frame_sequence)} frames, expected {self.frames_per_sequence}")
            # Pad with zeros if necessary or drop the sequence
            if len(frame_sequence) == 0:
                return False
            # Here we might choose to drop it, but for now let's just use what we have or pad
            padding = [np.zeros_like(frame_sequence[0])] * (self.frames_per_sequence - len(frame_sequence))
            frame_sequence.extend(padding)
            
        # Convert list to numpy array: [Time, Channels, Height, Width]
        sequence_array = np.stack(frame_sequence, axis=0)
        
        if output_filename is None:
            base_name = os.path.basename(video_path).split('.')[0]
            output_filename = f"{base_name}_mouth.npy"
            
        save_path = os.path.join(self.output_dir, output_filename)
        np.save(save_path, sequence_array)
        print(f"Saved processed sequence to {save_path} with shape {sequence_array.shape}")
        return True

if __name__ == "__main__":
    # Example usage:
    print("MouthCropper initialized. Please instantiate and call process_video(video_path).")
