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
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
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
