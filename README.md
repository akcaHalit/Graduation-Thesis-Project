# 🎣 Fish Counting and Length Estimation System

**Stereo Vision-Based Automated Fish Monitoring**

This project implements a computer vision pipeline for detecting, tracking, counting, and measuring fish in real time using stereo camera input. The system integrates YOLO11 object detection, ByteTrack multi-object tracking, and depth-based length estimation into a unified solution. A Flask web interface provides an accessible user experience for uploading and processing videos.

---

## 📋 Features

- 🎯 **Fish Detection:** Accurate fish and container box detection using YOLO11
- 🐟 **Tracking & Counting:** Identity-preserving counting logic with ByteTrack and dynamic ROI regions
- 📏 **Length Estimation:** Real-world length estimation via stereo disparity maps (two alternative methods)
- 🌐 **Web Interface:** Simple upload and visualization via Flask
- ⚡ **Real-Time Processing:** Optimized pipeline for efficient inference

---

## 🧠 Technical Overview

This project includes the following key components and experiments:

✅ **YOLO11-Based Object Detection**
- Two separate YOLO11 models were fine-tuned:
  - Fish detector trained on a custom dataset (S_aurata species)
  - Container box (crate) detector for defining counting regions
- Models achieved high detection performance (mAP@50 ~93%)

✅ **ByteTrack Multi-Object Tracking**
- Used for robust ID assignment and temporal consistency
- Preserves object identities even with occlusions and low-confidence detections
- Ensures each fish is counted exactly once when crossing the ROI

✅ **Stereo Vision & Depth Processing**
- Stereo camera inputs generate disparity maps for depth estimation
- Two length estimation methods implemented:
  1. **3D Point Transformation:**
     - Reconstruct 3D coordinates of bounding box corners
     - Compute Euclidean distance in real-world units
  2. **Bounding Box Scaling:**
     - Estimate size based on 2D width and mean depth scaling
- Trade-offs between accuracy and performance were evaluated

✅ **Flask Web Interface**
- Minimal interface to upload videos and visualize processed results

---

### ⚠️ Challenges Encountered

- **Stereo Disparity Noise:**
  - Low resolution and motion blur led to inaccurate depth maps
  - Small fish size increased disparity errors

- **Tracking Stability:**
  - 2D pipeline worked reliably
  - 3D pipeline struggled with maintaining consistent IDs

- **Generalization:**
  - The system was trained on a single fish species
  - Performance on multi-species scenarios may degrade

- **Performance Trade-offs:**
  - High-precision length estimation required accurate calibration and higher GPU resources
  - Faster estimation methods had larger error margins

---

### 💡 Future Improvements

- Use more robust stereo matching algorithms (e.g., StereoSGBM)
- Train on multi-species, more diverse datasets
- Experiment with DeepSORT or Transformer-based trackers
- Aggregate length estimates over the track lifetime rather than per-frame

---

## 🗂️ Project Directory Structure

```text
Graduation-Thesis-Project/
├── models/            # Trained YOLO11 weights (fish and container detectors)
├── scripts/           # Training, inference, and length estimation scripts
├── app/               # Flask web application
├── data/              # Sample datasets and test videos
├── utils/             # Helper modules and functions
├── requirements.txt   # Python dependency list
└── README.md          # Project documentation
