# ── config.py ──────────────────────────────────────────────────────────────
CAMERA_INDEX = 0  # 0 = default webcam; use RTSP URL for IP cam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

MODEL_NAME = "buffalo_l"  # InsightFace model: buffalo_l (best), buffalo_s (fast)
CTX_ID = 0  # 0 = GPU if available, -1 = force CPU
DET_SIZE = (640, 640)  # Detection input size; reduce to (320,320) for speed

MATCH_THRESHOLD = 0.55  # Cosine similarity threshold (0–1); tune this
TOP_K = 1  # Return best match per face

DB_EMBEDDINGS_PATH = r"database\embeddings.npy"
DB_LABELS_PATH = r"database\labels.json"

ALERT_SOUND = True
DRAW_LANDMARKS = False  # Set True to draw 5 facial landmarks
BOX_COLOR_MATCH = (0, 0, 255)  # Red for criminal match
BOX_COLOR_UNKNOWN = (0, 255, 0)  # Green for unknown
FONT = 1  # cv2.FONT_HERSHEY_COMPLEX
