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
        if os.path.exists(config.DB_EMBEDDINGS_PATH) and os.path.exists(
            config.DB_LABELS_PATH
        ):
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
        sims = self.embeddings @ query  # shape (N,)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= config.MATCH_THRESHOLD:
            return self.labels[best_idx], best_score
        return "Unknown", best_score
