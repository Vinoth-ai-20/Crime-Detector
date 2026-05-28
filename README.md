# Crime Detector — Real-Time Criminal Face Detection System

A FastAPI-based real-time facial recognition system designed to detect and identify criminals from live camera feeds and uploaded videos.

## Features

- **Real-Time Detection**: Live webcam feed analysis with instant criminal identification
- **Evidence Analysis**: Process recorded videos for suspect matching
- **Database Integration**: Persistent criminal database with face embeddings
- **REST API**: Full-featured API for integration with other systems
- **Web Dashboard**: Beautiful dark-themed UI for easy monitoring
- **Batch Training**: Auto-enroll criminals from dataset folder

## Tech Stack

| Component | Library |
|-----------|---------|
| Face Detection | InsightFace (buffalo_l model) |
| Face Recognition | ArcFace embeddings (512-dimensional) |
| Camera Input | OpenCV |
| Web Framework | FastAPI |
| Server | Uvicorn |
| Frontend | HTML5 + Vanilla JS |

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Database from Dataset

The system automatically enrolls all criminals from the `dataset/` folder structure:

```
dataset/
  criminal_1/
    img1.jpg, img2.jpg, ...
  criminal_2/
    img1.jpg, img2.jpg, ...
```

Run training:

```bash
python train.py
```

This will:

- Process all images in each criminal's folder
- Extract face embeddings using InsightFace
- Average multiple images per criminal
- Save to `database/embeddings.npy` and `database/labels.json`

## Usage

### Start the Server

```bash
python app.py
# or
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Then visit: **<http://localhost:8000>**

### Live Detection

1. Click "Live Camera Feed" button
2. System will display real-time face detection with criminal matches highlighted in red
3. Unknown faces shown in green
4. Alerts trigger when a wanted criminal is detected

### Video Analysis

1. Click "Upload Video Evidence"
2. Select a video file (MP4, AVI, MOV, MKV, WebM)
3. Click "Analyze Video" to process
4. System samples frames and reports detections

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/video_feed` | GET | Live video stream (MJPEG) |
| `/latest_detection` | GET | Latest criminal detection |
| `/detections` | GET | Recent detections history |
| `/upload` | POST | Upload video for analysis |
| `/criminal/{id}` | GET | Criminal information |
| `/criminals` | GET | List all criminals |
| `/stats` | GET | System statistics |
| `/enroll` | POST | Enroll new criminal |

### Example API Calls

**Get latest detection:**

```bash
curl http://localhost:8000/latest_detection
```

**List all criminals:**

```bash
curl http://localhost:8000/criminals
```

**Get system stats:**

```bash
curl http://localhost:8000/stats
```

**Enroll new criminal:**

```bash
curl -X POST -F "name=John Doe" -F "file=@photo.jpg" \
  http://localhost:8000/enroll
```

## Configuration

Edit `config.py` to customize:

```python
CAMERA_INDEX        = 0          # Webcam index (0 = default)
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720
MODEL_NAME          = "buffalo_l"  # or "buffalo_s" for speed
MATCH_THRESHOLD     = 0.55         # Cosine similarity threshold (0-1)
ALERT_SOUND         = True
```

### Tuning Match Threshold

| Threshold | Behavior |
|-----------|----------|
| 0.45 | More sensitive (fewer misses, more false positives) |
| 0.55 | **Balanced (recommended)** |
| 0.65 | Very strict (fewer false positives) |

## Directory Structure

```
Crime-detector/
├── app.py                 # FastAPI server
├── train.py              # Batch training script
├── config.py             # Configuration
├── enroll.py             # Single criminal enrollment
├── main.py               # Legacy CLI interface
├── requirements.txt
├── utils/
│   ├── detector.py       # Face detection wrapper
│   ├── database.py       # Embedding storage
│   └── alert.py          # Alert system
├── dataset/              # Training images (organized by criminal)
├── database/             # Persistent criminal database
│   ├── embeddings.npy
│   └── labels.json
├── models/               # ONNX model files (auto-downloaded)
├── static/               # Static files
├── templates/            # HTML interface
│   └── index.html
└── venv/                 # Virtual environment
```

## Performance

| Metric | Value |
|--------|-------|
| Detection Speed (buffalo_l) | ~25 FPS (CPU) / ~90 FPS (GPU) |
| Recognition Accuracy | ~99.7% (LFW dataset) |
| Embedding Dimension | 512-D (ArcFace) |
| Database Size | Unlimited (using FAISS for large DBs) |

## Troubleshooting

### "Camera not available"

- Check if webcam is connected
- Verify `CAMERA_INDEX` in config.py (try 0, 1, 2, etc.)
- Grant camera permissions to Python

### "Low FPS"

- Switch to `buffalo_s` model (faster, slightly less accurate)
- Reduce `DET_SIZE` to (320, 320)
- Use GPU with `onnxruntime-gpu`

### "No face detected"

- Improve lighting
- Ensure face is at least 80x80 pixels
- Move closer to camera

### "False positives"

- Increase `MATCH_THRESHOLD` to 0.60-0.65
- Enroll more varied angles of each criminal

## License

This project is for law enforcement and security purposes only.

## Support

For issues, check the logs or run with debug mode:

```bash
uvicorn app:app --reload --log-level debug
```
