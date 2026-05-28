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
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30), config.FONT, 0.8, (255, 255, 0), 2
        )
        cv2.putText(
            frame, f"Faces: {len(faces)}", (10, 60), config.FONT, 0.8, (255, 255, 0), 2
        )

        cv2.imshow("Crime Detector — Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
