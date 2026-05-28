# ── utils/alert.py ────────────────────────────────────────────────────────────
import winsound
import threading
import config


def trigger_alert(name: str, score: float):
    """Non-blocking audio + console alert on criminal match."""
    print(f"[⚠ ALERT] Match found: {name} | Confidence: {score:.2%}")
    if config.ALERT_SOUND:
        threading.Thread(target=winsound.Beep, args=(1000, 500), daemon=True).start()
