# Lesion Based Video Detection

A real-time **AI-powered lesion detection system** for medical videos (e.g., endoscopy, colonoscopy, surgical videos) using Faster R-CNN.

## ✨ Features

- Upload and process medical videos (MP4, AVI, MOV)
- Real-time lesion detection using **Faster R-CNN (ResNet50)** 
- Side-by-side comparison:
  - Left: Original Video (25 FPS)
  - Right: Processed Video with bounding boxes
- Frame-by-frame synchronization
- GPU acceleration support
- Detailed logging and error handling

## 🛠 Tech Stack

- **Backend**: Flask, OpenCV, PyTorch
- **Model**: Faster R-CNN (ResNet50 FPN) fine-tuned for lesion detection
- **Frontend**: HTML + JavaScript (synchronized dual video streaming)
- **Deployment**: CPU / NVIDIA GPU

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended)
- Pre-trained model: `lesion_detector.pth`

### Installation

```bash
git clone <your-repo-url>
cd lesion-video-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
