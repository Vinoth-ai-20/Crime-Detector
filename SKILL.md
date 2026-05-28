---
name: crime-detector
description: >
  Real-time criminal face detection system using live camera feed. Detects faces
  using InsightFace (ArcFace backbone) and matches them against a registered criminal
  database using cosine similarity on 512-d embeddings. Triggers include: live camera
  face detection, face enrollment, database matching, alert generation, and identity
  confidence scoring.
project_dir: E:\projects\Crime-detector
language: Python 3.10+
---

# Crime Detector — Real-Time Face Detection & Recognition Skill

## Project Overview

A real-time surveillance system that:
1. Captures live frames from a webcam or IP camera
2. Detects all faces in each frame using a fast, accurate model
3. Extracts 512-dimensional face embeddings
4. Matches embeddings against a criminal database using cosine similarity
5. Overlays bounding boxes, identity labels, and confidence scores on the frame
6. Fires alerts when a match exceeds a confidence threshold

---

## Recommended Tech Stack

| Layer | Library | Why |
|---|---|---|
| Face Detection | **InsightFace** (`buffalo_l` model) | Best accuracy/speed balance; ArcFace backbone; ONNX runtime |
| Face Recognition | **InsightFace** FaceAnalysis | Same pipeline handles detection + embedding in one call |
| Camera Input | **OpenCV** (`cv2`) | Industry standard; cross-platform; fast frame capture |
| Embedding Storage | **NumPy `.npy` + JSON** (small DB) or **FAISS** (large DB) | FAISS enables sub-millisecond similarity search at scale |
| Alert System | **Playsound / winsound** + **OpenCV overlay** | Simple audio + visual alert |
| Optional GPU | **ONNX Runtime with CUDAExecutionProvider** | Boosts FPS on NVIDIA GPUs |

---

## Project Directory Structure

```
E:\projects\Crime-detector\
├── main.py                  # Entry point — live camera loop
├── enroll.py                # Add a new criminal face to the database
├── database\
│   ├── embeddings.npy       # (N, 512) float32 array of all enrolled face embeddings
│   ├── labels.json          # ["Name1", "Name1", "Name2", ...] — one per embedding
│   └── photos\              # Optional: original enrollment photos
├── models\                  # InsightFace auto-downloads here; or place custom ONNX
├── utils\
│   ├── detector.py          # FaceAnalyzer wrapper class
│   ├── database.py          # Load, save, search embeddings
│   └── alert.py             # Trigger visual + audio alerts
├── config.py                # All tunable constants
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows

# Install dependencies
pip install insightface onnxruntime opencv-python numpy faiss-cpu
# For GPU: pip install onnxruntime-gpu  (replace onnxruntime)
```

**`requirements.txt`:**
```
insightface>=0.7.3
onnxruntime>=1.17.0          # or onnxruntime-gpu
opencv-python>=4.9.0
numpy>=1.26.0
faiss-cpu>=1.8.0             # optional for large DB
```

---

## config.py

```python
# ── config.py ──────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0          # 0 = default webcam; use RTSP URL for IP cam
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720
FPS_TARGET          = 30

MODEL_NAME          = "buffalo_l"  # InsightFace model: buffalo_l (best), buffalo_s (fast)
CTX_ID              = 0            # 0 = GPU if available, -1 = force CPU
DET_SIZE            = (640, 640)   # Detection input size; reduce to (320,320) for speed

MATCH_THRESHOLD     = 0.55         # Cosine similarity threshold (0–1); tune this
TOP_K               = 1            # Return best match per face

DB_EMBEDDINGS_PATH  = r"database\embeddings.npy"
DB_LABELS_PATH      = r"database\labels.json"

ALERT_SOUND         = True
DRAW_LANDMARKS      = False        # Set True to draw 5 facial landmarks
BOX_COLOR_MATCH     = (0, 0, 255)  # Red for criminal match
BOX_COLOR_UNKNOWN   = (0, 255, 0)  # Green for unknown
FONT                = 1            # cv2.FONT_HERSHEY_COMPLEX
```

---

## utils/detector.py — Face Analyzer Wrapper

```python
# ── utils/detector.py ───────────────────────────────────────────────────────
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import config

class FaceAnalyzer:
    """
    Wraps InsightFace FaceAnalysis for detection + embedding extraction.
    Uses ArcFace (buffalo_l) for state-of-the-art face recognition.
    """

    def __init__(self):
        self.app = FaceAnalysis(
            name=config.MODEL_NAME,
            root="models",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=config.CTX_ID, det_size=config.DET_SIZE)

    def get_faces(self, frame: np.ndarray) -> list:
        """
        Detect all faces in a BGR frame.
        Returns list of Face objects, each with:
          .bbox      → [x1, y1, x2, y2]
          .embedding → np.ndarray shape (512,)
          .kps       → 5 keypoints (optional)
          .det_score → detection confidence
        """
        return self.app.get(frame)
```

---

## utils/database.py — Embedding Store & Cosine Search

```python
# ── utils/database.py ────────────────────────────────────────────────────────
import numpy as np
import json
import os
import config

class FaceDatabase:
    """
    Manages enrolled criminal embeddings.
    Supports flat cosine search (small DB) or FAISS (large DB).
    """

    def __init__(self):
        self.embeddings: np.ndarray | None = None  # shape (N, 512)
        self.labels: list[str] = []
        self._load()

    def _load(self):
        if os.path.exists(config.DB_EMBEDDINGS_PATH) and os.path.exists(config.DB_LABELS_PATH):
            self.embeddings = np.load(config.DB_EMBEDDINGS_PATH)
            with open(config.DB_LABELS_PATH, "r") as f:
                self.labels = json.load(f)
            print(f"[DB] Loaded {len(self.labels)} enrolled faces.")
        else:
            print("[DB] No database found. Enroll faces first with enroll.py")

    def save(self):
        os.makedirs("database", exist_ok=True)
        np.save(config.DB_EMBEDDINGS_PATH, self.embeddings)
        with open(config.DB_LABELS_PATH, "w") as f:
            json.dump(self.labels, f)

    def add(self, name: str, embedding: np.ndarray):
        """Enroll a new face. Normalizes embedding before storing."""
        emb = embedding / np.linalg.norm(embedding)
        if self.embeddings is None:
            self.embeddings = emb[np.newaxis, :]
        else:
            self.embeddings = np.vstack([self.embeddings, emb[np.newaxis, :]])
        self.labels.append(name)
        self.save()

    def search(self, embedding: np.ndarray) -> tuple[str, float]:
        """
        Returns (label, cosine_similarity) for the closest match.
        Returns ("Unknown", 0.0) if DB is empty or no match found.
        """
        if self.embeddings is None or len(self.labels) == 0:
            return "Unknown", 0.0

        query = embedding / np.linalg.norm(embedding)
        # Cosine similarity = dot product of L2-normalized vectors
        sims = self.embeddings @ query          # shape (N,)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= config.MATCH_THRESHOLD:
            return self.labels[best_idx], best_score
        return "Unknown", best_score
```

---

## utils/alert.py — Alert System

```python
# ── utils/alert.py ────────────────────────────────────────────────────────────
import winsound
import threading
import config

def trigger_alert(name: str, score: float):
    """Non-blocking audio + console alert on criminal match."""
    print(f"[⚠ ALERT] Match found: {name} | Confidence: {score:.2%}")
    if config.ALERT_SOUND:
        threading.Thread(
            target=winsound.Beep, args=(1000, 500), daemon=True
        ).start()
```

---

## main.py — Live Detection Loop

```python
# ── main.py ──────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import time
import config
from utils.detector import FaceAnalyzer
from utils.database import FaceDatabase
from utils.alert import trigger_alert

def draw_face(frame, face, label: str, score: float):
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    is_match = label != "Unknown"
    color = config.BOX_COLOR_MATCH if is_match else config.BOX_COLOR_UNKNOWN

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label}  {score:.0%}" if is_match else "Unknown"
    (tw, th), _ = cv2.getTextSize(text, config.FONT, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4), config.FONT, 0.6, (255, 255, 255), 1)

    if config.DRAW_LANDMARKS and face.kps is not None:
        for kp in face.kps.astype(int):
            cv2.circle(frame, tuple(kp), 2, (0, 255, 255), -1)


def main():
    analyzer = FaceAnalyzer()
    db = FaceDatabase()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {config.CAMERA_INDEX}")

    print("[INFO] Starting live detection. Press 'q' to quit.")
    alerted_ids: set[str] = set()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed. Retrying...")
            continue

        faces = analyzer.get_faces(frame)

        for face in faces:
            label, score = db.search(face.embedding)
            draw_face(frame, face, label, score)

            # Avoid repeated alerts for the same person in the same session
            if label != "Unknown" and label not in alerted_ids:
                trigger_alert(label, score)
                alerted_ids.add(label)

        # FPS overlay
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    config.FONT, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                    config.FONT, 0.8, (255, 255, 0), 2)

        cv2.imshow("Crime Detector — Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## enroll.py — Add Criminal to Database

```python
# ── enroll.py ────────────────────────────────────────────────────────────────
"""
Usage:
  python enroll.py --name "John Doe" --image path\to\photo.jpg
  python enroll.py --name "Jane Doe" --camera          # capture from webcam
"""
import argparse
import cv2
import sys
from utils.detector import FaceAnalyzer
from utils.database import FaceDatabase

def enroll_from_image(analyzer, db, name: str, image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)
    faces = analyzer.get_faces(frame)
    if not faces:
        print("[ERROR] No face detected in the image.")
        sys.exit(1)
    if len(faces) > 1:
        print(f"[WARN] {len(faces)} faces found; using the largest.")
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    db.add(name, faces[0].embedding)
    print(f"[OK] Enrolled '{name}' successfully. DB size: {len(db.labels)}")


def enroll_from_camera(analyzer, db, name: str):
    cap = cv2.VideoCapture(0)
    print("[INFO] Press SPACE to capture, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Enroll — Press SPACE", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            faces = analyzer.get_faces(frame)
            if not faces:
                print("[WARN] No face detected. Try again.")
                continue
            db.add(name, faces[0].embedding)
            print(f"[OK] Enrolled '{name}'. DB size: {len(db.labels)}")
            break
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Criminal's full name")
    parser.add_argument("--image", default=None, help="Path to photo")
    parser.add_argument("--camera", action="store_true", help="Capture from webcam")
    args = parser.parse_args()

    analyzer = FaceAnalyzer()
    db = FaceDatabase()

    if args.image:
        enroll_from_image(analyzer, db, args.name, args.image)
    elif args.camera:
        enroll_from_camera(analyzer, db, args.name)
    else:
        print("[ERROR] Provide --image or --camera")
        sys.exit(1)
```

---

## Copilot Usage Patterns

When asking GitHub Copilot for help inside this project, use these prompt patterns:

| Goal | Prompt to Copilot |
|---|---|
| Improve match speed | `# Use FAISS IndexFlatIP for large-scale embedding search` |
| Multiple cameras | `# Add multi-camera support using threading, one thread per cap` |
| Log matches to CSV | `# Log match events: timestamp, name, confidence, frame snapshot path` |
| RTSP / IP cam | `# Replace CAMERA_INDEX with RTSP URL for IP camera stream` |
| Send email alert | `# Send email alert using smtplib when criminal is detected` |
| Export snapshot | `# Save cropped face image to alerts/ folder on each match` |
| Batch enrollment | `# Enroll all images in a folder, using filename as the label` |
| Face anti-spoofing | `# Add liveness detection using Silent-Face-Anti-Spoofing model` |

---

## Model Performance Reference

| Model | Detection Speed | Recognition Accuracy (LFW) | RAM Usage |
|---|---|---|---|
| `buffalo_s` | ~60 FPS (CPU) | ~99.2% | ~120 MB |
| `buffalo_l` | ~25 FPS (CPU) | **~99.7%** | ~320 MB |
| `buffalo_l` + CUDA | ~90 FPS (GPU) | ~99.7% | ~600 MB VRAM |

> **Recommended default:** `buffalo_l` on CPU for accuracy; switch to `buffalo_s` if FPS is below 15.

---

## Tuning `MATCH_THRESHOLD`

| Threshold | Behavior |
|---|---|
| `0.45` | More sensitive — fewer missed detections, more false positives |
| `0.55` | **Balanced (recommended default)** |
| `0.65` | Very strict — fewer false positives, may miss low-quality matches |

Tune this value using test images before deployment. Lower threshold in poor lighting.

---

## Common Issues & Fixes

| Problem | Fix |
|---|---|
| `No module named insightface` | `pip install insightface onnxruntime` |
| Low FPS on CPU | Switch to `buffalo_s`; reduce `DET_SIZE` to `(320,320)` |
| No face detected | Improve lighting; ensure face is >80px wide in frame |
| ONNX CUDA error | Install matching `onnxruntime-gpu` for your CUDA version |
| False matches | Increase `MATCH_THRESHOLD`; enroll more angles per person |
| Camera not opening | Check `CAMERA_INDEX`; use `cap = cv2.VideoCapture("rtsp://...")` for IP cam |

---

## Notes for Copilot

- All face embeddings are **L2-normalized before storage and search**. Do not skip normalization.
- `FaceAnalysis.get()` returns faces sorted by detection confidence descending.
- Always call `analyzer.get_faces(frame)` on the **original BGR frame** from OpenCV; do not resize before detection.
- Use `face.det_score` to filter out low-confidence detections (e.g., skip faces with `det_score < 0.7`).
- The `database/` folder is **gitignored** by convention — never commit real criminal data.
