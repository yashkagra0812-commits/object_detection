# Real-Time Object Detection

A complete, self-hosted real-time object detection system: a Python/Flask
backend runs a YOLOv8 model (via `ultralytics`, on top of PyTorch and
OpenCV) against a live webcam feed, and a polished HTML/CSS/JS frontend
displays the annotated video plus a live list of detected objects.

The dashboard now also includes a Start/Stop session button so you can pause
and resume the detection workflow at any time.

```
realtime-object-detection/
├── app.py              # Flask app: routes, video streaming endpoint, JSON APIs
├── camera.py            # Threaded camera capture (keeps video smooth)
├── detector.py           # Threaded YOLOv8 inference + bounding-box drawing
├── config.py             # All tunable settings in one place
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Frontend page
└── static/
    ├── css/style.css       # Styling
    └── js/script.js         # Polls the backend and updates the UI
```

## 1. How it works (architecture)

The hardest part of "real-time" object detection is that camera capture is
fast (a webcam can usually deliver 30 FPS) but neural-network inference is
comparatively slow (a CPU might only manage 10-20 FPS with a small model).
If you naively do "grab frame → run model → display frame" in a single
loop, your video's frame rate collapses to the model's frame rate, and it
feels laggy.

This project avoids that by running **three independent loops**, each in
its own thread, so a slow model never blocks the video:

| Thread | File | Job | Speed |
|---|---|---|---|
| **Capture** | `camera.py` → `CameraStream` | Continuously reads frames from the webcam and keeps only the newest one | Camera's native FPS (e.g. ~30) |
| **Detection** | `detector.py` → `DetectorThread` | Takes whatever the newest camera frame is, runs YOLOv8 on it, stores the resulting boxes/labels | Model's sustainable FPS (e.g. ~15-20 on CPU) |
| **Streaming** | `app.py` → `/video_feed` | On each tick, takes the newest camera frame, draws the *most recent* detection results on top, JPEG-encodes it, and sends it to the browser | Capped at `TARGET_STREAM_FPS` |

Because these three loops are decoupled, the video you see in the browser
stays smooth and responsive (bounded by camera + JPEG encode speed) even
though the bounding boxes update at whatever rate the model can keep up
with. On fast motion you may notice boxes lag the video by a frame or two
— that's the intentional trade-off that keeps the *video* itself
real-time.

### Backend ↔ Frontend integration

- **Video**: `/video_feed` is an MJPEG stream
  (`multipart/x-mixed-replace`). The frontend just points an `<img>` tag at
  it — the browser natively renders each new JPEG frame as it arrives, no
  custom JavaScript required for the video itself.
- **Detection data**: `/api/detections` returns the current list of
  detected objects as JSON (`label`, `confidence`, `box`). `script.js`
  polls this every 500 ms to populate the "Detected Objects" sidebar.
- **Performance stats**: `/api/stats` returns camera FPS, inference FPS,
  inference latency (ms), and object count, also polled every 500 ms and
  shown in the stats bar above the video.

The bounding boxes visible *in the video* are drawn server-side (by
`DetectorThread.draw()`) directly onto the JPEG frames, which guarantees
the boxes are always aligned with the frame they're drawn on. The sidebar
list is a separate, independent JSON view of the same underlying detection
data, useful for showing labels/confidence in a clean list format.

## 2. Setup

### Prerequisites
- Python 3.9+
- A webcam accessible to your OS or a phone IP webcam stream URL
- (Optional) an NVIDIA GPU + CUDA-enabled PyTorch for faster inference

### Install

```bash
cd realtime-object-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first time you run the app, `ultralytics` will automatically download
the `yolov8n.pt` model weights (~6 MB).

### Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser. You should see the
live camera feed with bounding boxes drawn around detected objects, an FPS
dashboard, and a running list of currently-detected object labels.

### Use a phone IP webcam

1. Install an IP webcam app on your phone, such as `DroidCam`, `IP Webcam`,
   or any app that exposes an MJPEG/RTSP stream over your Wi-Fi network.
2. Start the app and note the stream URL, for example:
   `http://192.168.1.100:8080/video`.
3. Open `config.py` and set:

```python
CAMERA_SOURCE = "http://192.168.1.100:8080/video"
```

4. Restart the app with `python app.py`.

If the stream is not opening, verify the phone and PC are on the same Wi-Fi
network and that the URL is accessible from your browser.

## 3. Tuning performance

All of the following live in `config.py`:

| Setting | Effect |
|---|---|
| `FRAME_WIDTH` / `FRAME_HEIGHT` | Lower resolution = higher FPS, less accurate localization of small objects |
| `MODEL_NAME` | `yolov8n.pt` (nano, fastest) → `yolov8s.pt`/`yolov8m.pt` (slower, more accurate) |
| `CONFIDENCE_THRESHOLD` | Raise to show only high-confidence detections (fewer false positives) |
| `DEVICE` | Set to `"cuda"` explicitly if you have a GPU and want to force GPU inference |
| `JPEG_QUALITY` | Lower = smaller/faster stream, more visible compression artifacts |
| `TARGET_STREAM_FPS` | Caps how fast the server pushes frames, regardless of camera speed |

**General tips for higher frame rates:**
1. Use the nano model (`yolov8n.pt`) unless you specifically need more
   accuracy — it's the single biggest lever for CPU-only speed.
2. Lower the capture resolution (`FRAME_WIDTH`/`FRAME_HEIGHT`) — YOLO
   resizes internally anyway, so a smaller source frame speeds up both
   capture and inference with only a modest accuracy cost.
3. If you have a discrete NVIDIA GPU, install a CUDA build of PyTorch and
   set `DEVICE = "cuda"` — inference speed typically improves 5-10x.
4. Reduce `JPEG_QUALITY` if network bandwidth (e.g. streaming over Wi-Fi
   to another device) is the bottleneck rather than inference.

## 4. Extending the project

- **Different model families**: `detector.py` only touches the
  `ultralytics.YOLO` API surface (`.predict()`, `.names`, `box.cls` /
  `box.conf` / `box.xyxy`), so swapping in any other YOLOv8/v9/v10
  checkpoint is a one-line change to `MODEL_NAME`.
- **OpenCV DNN / other frameworks**: if you'd rather not depend on
  PyTorch, `detector.py` is the only file that would need to change —
  replace the `YOLO(...)` model with `cv2.dnn.readNet(...)` and adapt the
  post-processing loop to your model's output format. `camera.py`,
  `app.py`, and the entire frontend are framework-agnostic.
- **Multiple cameras**: instantiate more than one `CameraStream` /
  `DetectorThread` pair and add a second `/video_feed_2` route.
- **Recording**: add a `cv2.VideoWriter` inside `_mjpeg_generator()` to
  save annotated frames to disk alongside streaming them.

## 5. Troubleshooting

- **"Could not open camera at index 0"** — another application may be
  using the webcam, or you may need to change `CAMERA_INDEX` in
  `config.py` if you have multiple cameras.
- **Video feels choppy** — check the `Camera FPS` and `Inference FPS`
  stats in the UI. If camera FPS is low, it's a hardware/driver issue
  unrelated to the model. If inference FPS is low but camera FPS is fine,
  see the performance tuning section above.
- **High CPU usage** — expected for CPU-based inference; lower the
  resolution or switch to a GPU per the tuning tips abo