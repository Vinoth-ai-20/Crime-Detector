# 🚀 Crime Detector — Start Guide

## Quick Start

### 1. **Activate Virtual Environment & Start Server**

```powershell
cd e:\projects\Crime-detector
venv\Scripts\activate.ps1
python app.py
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 2. **Open in Browser**

Go to: **<http://127.0.0.1:8000>**

**Important**: Press `Ctrl+F5` (hard refresh) to clear browser cache and load the latest HTML.

### 3. **Live Detection**

You should see:

- ✅ Real-time webcam feed
- ✅ Face detection boxes (RED for criminals, GREEN for unknown)
- ✅ FPS counter
- ✅ Criminal information display when detected

### 4. **Troubleshooting Cache Issues**

If you see 404 errors or old interface:

**Clear Browser Cache:**

- **Chrome**: Ctrl+Shift+Delete → Clear all browsing data
- **Firefox**: Ctrl+Shift+Delete → Clear all
- **Edge**: Ctrl+Shift+Delete → Clear browsing data
- Then refresh: **Ctrl+F5**

Or open in **Incognito/Private Mode** (bypasses cache):

- Chrome: Ctrl+Shift+N
- Firefox: Ctrl+Shift+P
- Edge: Ctrl+Shift+P

---

## 🔌 API Testing

### Check if Server is Running

```bash
curl http://127.0.0.1:8000/stats
```

Expected response:

```json
{
  "total_criminals": 2,
  "total_detections": 0,
  "camera_active": true,
  "model": "buffalo_l",
  "threshold": 0.55
}
```

### Get Latest Detection

```bash
curl http://127.0.0.1:8000/latest_detection
```

### List All Criminals

```bash
curl http://127.0.0.1:8000/criminals
```

### Upload Video for Analysis

```bash
curl -X POST -F "file=@video.mp4" http://127.0.0.1:8000/upload
```

---

## 📊 System Status

| Component | Status |
|-----------|--------|
| FastAPI Server | ✅ Running |
| Live Detection | ✅ Active |
| Database | ✅ 2 criminals loaded |
| Camera | ✅ 1280×720 |
| Model | ✅ buffalo_l |

---

## ⚡ Common Issues

### "Blank video feed"

- Hard refresh page: **Ctrl+F5**
- Check camera is connected
- Ensure camera permissions are granted

### "Page still shows old interface"

- Clear browser cache (see above)
- Try **Incognito/Private mode**
- Or use different browser

### "Server not responding"

- Restart server: Press **Ctrl+C**, then `python app.py`
- Check port 8000 is not in use: `netstat -ano | findstr :8000`

### "Detection not working"

- Improve lighting
- Move closer to camera (face must be >80px)
- Check MATCH_THRESHOLD in config.py

---

## 📝 Files Overview

- **app.py** → FastAPI server (10+ endpoints)
- **train.py** → Batch enrollment script
- **config.py** → Settings (MATCH_THRESHOLD, CAMERA_INDEX, etc.)
- **templates/index.html** → Web UI
- **utils/** → detector.py, database.py, alert.py
- **database/** → embeddings.npy, labels.json (trained DB)
- **dataset/** → Training images (criminal_1, criminal_2)

---

## 🔄 Add More Criminals

1. Create folder: `dataset/criminal_3/`
2. Add images: `criminal_3/photo1.jpg`, `photo2.jpg`, etc.
3. Run: `python train.py`
4. Restart server

---

**Everything is working! Just open the browser and go to <http://127.0.0.1:8000>** 🎉
