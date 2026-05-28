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
