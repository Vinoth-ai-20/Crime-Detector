import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from utils.detector import FaceAnalyzer
from utils.database import FaceDatabase
import config

"""
Training script to batch enroll all criminals from the dataset folder.
Dataset structure expected:
  dataset/
    criminal_1/
      img1.jpg
      img2.jpg
      ...
    criminal_2/
      img1.jpg
      ...
"""


def train_from_dataset():
    """Enroll all criminals from dataset folder."""
    dataset_path = Path("dataset")

    if not dataset_path.exists():
        print("[ERROR] dataset folder not found!")
        sys.exit(1)

    analyzer = FaceAnalyzer()
    db = FaceDatabase()

    print(f"\n[INFO] Starting batch enrollment from {dataset_path}")
    print(f"[INFO] Current database size: {len(db.labels)} criminals")

    enrolled_count = 0
    failed_count = 0

    # Iterate through each criminal folder
    for criminal_folder in sorted(dataset_path.iterdir()):
        if not criminal_folder.is_dir():
            continue

        criminal_name = criminal_folder.name
        print(f"\n[PROCESSING] Criminal: {criminal_name}")

        criminal_embeddings = []

        # Process each image in the criminal's folder
        for image_file in sorted(criminal_folder.glob("*.[jJ][pP]*")):
            print(f"  → Reading {image_file.name}...", end=" ")

            frame = cv2.imread(str(image_file))
            if frame is None:
                print("[SKIP] Cannot read image")
                failed_count += 1
                continue

            try:
                faces = analyzer.get_faces(frame)

                if not faces:
                    print("[SKIP] No face detected")
                    failed_count += 1
                    continue

                if len(faces) > 1:
                    # Use the largest face
                    faces = sorted(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                        reverse=True,
                    )
                    print(f"[INFO] {len(faces)} faces found, using largest")

                embedding = faces[0].embedding
                criminal_embeddings.append(embedding)
                print("[OK]")

            except Exception as e:
                print(f"[ERROR] {str(e)}")
                failed_count += 1
                continue

        # Average embeddings for this criminal
        if criminal_embeddings:
            avg_embedding = np.mean(criminal_embeddings, axis=0)
            db.add(criminal_name, avg_embedding)
            enrolled_count += 1
            print(
                f"[SUCCESS] Enrolled {criminal_name} ({len(criminal_embeddings)} images)"
            )
        else:
            print(f"[FAILED] No valid faces found for {criminal_name}")

    print(f"\n" + "=" * 60)
    print(f"[SUMMARY]")
    print(f"  Enrolled: {enrolled_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total criminals in DB: {len(db.labels)}")
    print(f"=" * 60 + "\n")


if __name__ == "__main__":
    train_from_dataset()
