# ── app.py ──────────────────────────────────────────────────────────────────
"""
FastAPI backend for Crime Detector system.
Provides REST API for live video detection, video upload, and criminal database.
Serves HTML frontend from templates/ folder.
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from collections import deque
from typing import Optional, List
from threading import Thread, Lock
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from utils.detector import FaceAnalyzer
from utils.database import FaceDatabase
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Crime Detector", version="1.0")

# Global state
analyzer: Optional[FaceAnalyzer] = None
db: Optional[FaceDatabase] = None
cap: Optional[cv2.VideoCapture] = None

# Detection state
latest_detection = {"id": None, "confidence": 0.0, "timestamp": 0}
detection_lock = Lock()
detection_queue = deque(maxlen=50)  # Keep last 50 detections
camera_thread = None
camera_running = False

# Criminal database (for API responses)
CRIMINAL_DB = {
    "criminal_1": {
        "name": "Criminal 1",
        "crime": "Robbery & Assault",
        "status": "Wanted",
        "reward": "$10,000",
        "last_seen": "May 25, 2026",
        "details": "Armed and dangerous. Proceed with caution.",
        "image": "/static/criminal_1.jpg",
    },
    "criminal_2": {
        "name": "Criminal 2",
        "crime": "Fraud & Theft",
        "status": "Fugitive",
        "reward": "$5,000",
        "last_seen": "May 24, 2026",
        "details": "High-level white collar criminal. Wanted in 3 states.",
        "image": "/static/criminal_2.jpg",
    },
}


@app.on_event("startup")
async def startup_event():
    """Initialize model and database on startup."""
    global analyzer, db, cap, camera_thread

    logger.info("Starting Crime Detector API...")

    try:
        analyzer = FaceAnalyzer()
        db = FaceDatabase()

        # Initialize camera
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not cap.isOpened():
            logger.warning(f"Camera {config.CAMERA_INDEX} not available")
            cap = None
        else:
            logger.info(f"Camera opened: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")

            # Start camera thread
            global camera_running
            camera_running = True
            camera_thread = Thread(target=camera_loop, daemon=True)
            camera_thread.start()

        logger.info("Crime Detector API started successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global camera_running, cap

    camera_running = False
    if cap:
        cap.release()

    logger.info("Crime Detector API shutdown")


def camera_loop():
    """Background thread for continuous camera detection."""
    logger.info("Camera thread started")
    alerted_ids = set()
    prev_time = time.time()
    frame_count = 0

    while camera_running:
        if cap is None or not cap.isOpened():
            time.sleep(1)
            continue

        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame capture failed")
            time.sleep(0.1)
            continue

        try:
            faces = analyzer.get_faces(frame)
            frame_count += 1

            for face in faces:
                label, score = db.search(face.embedding)

                if label != "Unknown" and score >= config.MATCH_THRESHOLD:
                    # Record detection
                    detection_data = {
                        "criminal": label,
                        "confidence": float(score),
                        "timestamp": time.time(),
                        "frame_number": frame_count,
                    }
                    detection_queue.append(detection_data)

                    # Update latest detection
                    with detection_lock:
                        latest_detection["id"] = label
                        latest_detection["confidence"] = score
                        latest_detection["timestamp"] = detection_data["timestamp"]

                    # Trigger alert once per session
                    if label not in alerted_ids:
                        logger.warning(
                            f"[ALERT] Criminal detected: {label} ({score:.1%})"
                        )
                        alerted_ids.add(label)

            # Log FPS every 30 frames
            if frame_count % 30 == 0:
                curr_time = time.time()
                fps = 30.0 / (curr_time - prev_time)
                prev_time = curr_time
                logger.debug(f"FPS: {fps:.1f} | Faces: {len(faces)}")

        except Exception as e:
            logger.error(f"Detection error: {e}")
            continue


def generate_frames():
    """Generator for video streaming."""
    if cap is None or not cap.isOpened():
        yield b""
        return

    alerted_ids = set()

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            faces = analyzer.get_faces(frame)

            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                label, score = db.search(face.embedding)

                # Determine color based on match
                is_match = label != "Unknown" and score >= config.MATCH_THRESHOLD
                color = config.BOX_COLOR_MATCH if is_match else config.BOX_COLOR_UNKNOWN

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                text = f"{label} {score:.0%}" if is_match else "Unknown"
                (tw, th), _ = cv2.getTextSize(text, config.FONT, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame, text, (x1 + 2, y1 - 4), config.FONT, 0.6, (255, 255, 255), 1
                )

                # Trigger alert
                if is_match and label not in alerted_ids:
                    alerted_ids.add(label)

            # Add FPS counter
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 30),
                config.FONT,
                0.8,
                (255, 255, 0),
                2,
            )

            # Encode frame
            ret, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + f"{len(frame_bytes)}".encode()
                + b"\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            continue


# ── API Routes ──


@app.get("/")
async def root():
    """Serve main HTML page."""
    return FileResponse("templates/index.html")


@app.get("/video_feed")
async def video_feed():
    """Live video streaming endpoint."""
    if cap is None:
        return JSONResponse({"error": "Camera not available"}, status_code=503)

    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/latest_detection")
async def latest_detection_endpoint():
    """Get latest detection."""
    with detection_lock:
        return latest_detection


@app.get("/detections")
async def get_detections(limit: int = 50):
    """Get recent detections."""
    return {"total": len(detection_queue), "detections": list(detection_queue)[-limit:]}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Process uploaded video for criminal detection."""
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    try:
        # Save uploaded file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process video
        detected_faces = []
        processed_frames = 0
        total_frames = 0

        cap_upload = cv2.VideoCapture(temp_path)
        if not cap_upload.isOpened():
            return JSONResponse({"error": "Cannot open video file"}, status_code=400)

        total_frames = int(cap_upload.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        # Process every Nth frame for speed
        sample_rate = max(1, total_frames // 50)  # Sample ~50 frames

        while True:
            ret, frame = cap_upload.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue

            try:
                faces = analyzer.get_faces(frame)

                for face in faces:
                    label, score = db.search(face.embedding)

                    if label != "Unknown" and score >= config.MATCH_THRESHOLD:
                        detected_faces.append(
                            {
                                "criminal": label,
                                "confidence": float(score),
                                "frame": frame_idx,
                            }
                        )

                processed_frames += 1

            except Exception as e:
                logger.error(f"Video processing error: {e}")
                continue

        cap_upload.release()

        # Clean up temp file
        import os

        os.remove(temp_path)

        return {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "detected_faces": detected_faces,
            "matches_found": len(detected_faces),
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/criminal/{criminal_id}")
async def get_criminal(criminal_id: str):
    """Get criminal information by ID."""
    if criminal_id in CRIMINAL_DB:
        return CRIMINAL_DB[criminal_id]

    # Return generic criminal data based on database label
    return {
        "name": criminal_id,
        "crime": "Unknown",
        "status": "Wanted",
        "reward": "Contact local authorities",
        "last_seen": "Unknown",
        "details": "Criminal identified by facial recognition system",
        "image": "/static/default.jpg",
    }


@app.get("/criminals")
async def list_criminals():
    """Get list of all tracked criminals."""
    return {"total": len(db.labels), "criminals": db.labels}


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "total_criminals": len(db.labels),
        "total_detections": len(detection_queue),
        "camera_active": cap is not None and cap.isOpened(),
        "model": config.MODEL_NAME,
        "threshold": config.MATCH_THRESHOLD,
    }


@app.post("/enroll")
async def enroll_criminal(name: str, file: UploadFile = File(...)):
    """Enroll a new criminal from an image."""
    try:
        # Read image
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        faces = analyzer.get_faces(frame)
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in image")

        # Use largest face
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        db.add(name, faces[0].embedding)

        return {"status": "success", "name": name, "total_criminals": len(db.labels)}

    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Static files mount failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
