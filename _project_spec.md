# Project Specification: SynchCheck (Multi-Modal Deepfake Detector)

## 1. Project Overview
**Objective:** Build a robust deepfake detection system that identifies forgeries based on the biological desynchronization between audio (phonemes) and visual mouth movements (visemes).
**Research Anchor:** "Joint Audio-Visual Deepfake Detection" (Zhou & Lim, ICCV 2021).
**Key Metric:** "Sync Distance" / Confidence score indicating the correlation between audio and video streams.

## 2. System Architecture (Siamese-style Fusion)
The system consists of two parallel feature extraction streams that converge into a fusion layer:

### A. Visual Stream (Viseme Encoder)
*   **Input:** Sequential frames of cropped mouth regions (Region of Interest).
*   **Architecture:** Pre-trained ResNet-18 or 3D-CNN (SlowFast) to capture temporal motion.
*   **Output:** Visual feature embedding vector.

### B. Audio Stream (Phoneme Encoder)
*   **Input:** Mel-frequency cepstral coefficients (MFCCs) extracted from the video’s audio track.
*   **Architecture:** ResNet-18 or a specialized 1D-CNN.
*   **Output:** Audio feature embedding vector.

### C. Fusion & Classification Layer
*   **Mechanism:** Cross-Modal Attention or simple Concatenation followed by a Multi-Layer Perceptron (MLP).
*   **Head:** Sigmoid activation to output a probability (0 = Real, 1 = Fake).

## 3. Technical Requirements & Stack
*   **Language:** Python 3.10+
*   **Deep Learning:** PyTorch + PyTorch Lightning (for organized training loops).
*   **Computer Vision:** MediaPipe (for face detection and mouth ROI cropping).
*   **Audio Signal Processing:** `librosa` (for MFCC extraction and resampling).
*   **Metadata/Config:** Project configuration and local dataset manifest MUST be stored in **XML format** to ensure structured, hierarchical metadata management.

## 4. MVP Implementation Goals (Weeks 1-3)
The AI Agent should prioritize these tasks to satisfy the Guide's initial review:
1.  **Data Pre-processing Pipeline:** 
    *   Script to extract audio from `.mp4`.
    *   Script to isolate 20-frame sequences of cropped mouth images.
    *   Save processed features as `.npy` or `.pt` tensors for fast loading.
2.  **Baseline Model Construction:** 
    *   Implement the dual-stream `DeepfakeDetector` class.
    *   Use Binary Cross-Entropy (BCE) loss for the initial classification task.
3.  **Local Training Loop:** 
    *   Functional training script that logs loss and accuracy to the console.

## 5. Coding Standards & Constraints
*   **Modularity:** Keep pre-processing, model definition, and training in separate modules.
*   **Device Agnostic:** Ensure tensors move to `cuda`, `mps`, or `cpu` automatically.
*   **Edge Case Handling:** Include logic to skip frames where a face is not detected.
*   **Documentation:** All functions must have docstrings explaining input/output shapes (e.g., `[Batch, Channels, Time, Height, Width]`).