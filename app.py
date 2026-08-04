"""
app.py - Real-Time Object Detection: Flask Backend
=====================================================

Routes
------
GET /                -> serves the HTML/CSS/JS frontend
GET /video_feed       -> MJPEG stream (multipart/x-mixed-replace) of camera
                          frames with live bounding boxes drawn on top
GET /api/detections    -> JSON array of currently detected objects
GET /api/stats         -> JSON camera FPS / inference FPS / latency / object count
GET /api/session       -> current session state
POST /api/session      -> start/stop the detector session
GET /api/session/report -> current session report metadata
GET /api/session/report/download -> downloadable CSV report
"""
import time
import urllib.parse

import cv2
from flask import Flask, Response, jsonify, make_response, render_template, request

import config

app = Flask(__name__)

# Initialize the camera and detector lazily so the app can be imported in
# environments such as Vercel without failing immediately on startup.
camera = None
detector = None
_runtime_error = None


def _initialize_runtime():
    global camera, detector, _runtime_error

    if camera is not None and detector is not None:
        return True

    try:
        from camera import CameraStream
        from detector import DetectorThread

        camera = CameraStream().start()
        time.sleep(0.5)
        detector = DetectorThread(camera).start()
        _runtime_error = None
        return True
    except Exception as exc:
        camera = None
        detector = None
        _runtime_error = str(exc)
        return False


def _mjpeg_generator():
    """Yield JPEG frames (with detections drawn on top) for the live preview."""
    if not _initialize_runtime():
        return

    frame_interval = 1.0 / config.TARGET_STREAM_FPS
    while True:
        loop_start = time.time()

        frame, _ = camera.read()
        if frame is not None:
            annotated = detector.draw(frame)
            ok, buffer = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
            )
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

        elapsed = time.time() - loop_start
        remaining = frame_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    if not _initialize_runtime():
        return Response(
            "Video stream unavailable: " + (_runtime_error or "camera startup failed"),
            status=503,
            mimetype="text/plain",
        )

    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _is_root_camera_url(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.netloc and parsed.path in ("", "/")


def _camera_url_candidates(url):
    normalized = url.strip().rstrip("/")
    if _is_root_camera_url(normalized):
        return [
            normalized,
            f"{normalized}/video",
            f"{normalized}/shot.jpg",
            f"{normalized}/mjpeg",
            f"{normalized}/stream",
            f"{normalized}/live",
            f"{normalized}/video.mjpeg",
            f"{normalized}/h264",
            f"{normalized}/cam.mjpeg",
        ]
    return [normalized]


@app.route("/api/connect_camera", methods=["POST"])
def api_connect_camera():
    if not _initialize_runtime():
        return jsonify({"error": _runtime_error or "Camera runtime is unavailable."}), 503

    data = request.get_json(silent=True) or {}
    url = data.get("url", "")
    if not isinstance(url, str) or not url.strip():
        return jsonify({"error": "Please provide a valid URL."}), 400

    normalized = url.strip()
    if not normalized.lower().startswith(("http://", "https://", "rtsp://")):
        return jsonify({"error": "URL must start with http://, https://, or rtsp://."}), 400

    candidates = _camera_url_candidates(normalized)
    last_error = None
    for candidate in candidates:
        try:
            camera.switch_source(candidate)
            return jsonify({"status": "ok", "url": candidate})
        except RuntimeError as exc:
            last_error = str(exc)

    return jsonify({"error": last_error or "Unable to connect to the IP camera."}), 400


@app.route("/api/detections")
def api_detections():
    if not _initialize_runtime():
        return jsonify({"error": _runtime_error or "Detector runtime is unavailable."}), 503

    detections, _, _, _ = detector.get_detections()
    return jsonify([
        {
            "label": d.label,
            "confidence": round(d.confidence, 3),
            "box": d.box,
        }
        for d in detections
    ])


@app.route("/api/stats")
def api_stats():
    if not _initialize_runtime():
        return jsonify({"error": _runtime_error or "Detector runtime is unavailable."}), 503

    _, camera_fps = camera.read()
    detections, inference_fps, inference_ms, status_message = detector.get_detections()
    report_info = detector.get_report_info()
    return jsonify({
        "camera_fps": round(camera_fps, 1),
        "inference_fps": round(inference_fps, 1),
        "inference_ms": round(inference_ms, 1),
        "object_count": len(detections),
        "session_active": detector.is_enabled(),
        "status_message": status_message,
        "report_available": report_info["available"],
        "report_filename": report_info["filename"],
        "report_summary": report_info["summary"],
        "report_total_detections": report_info["total_detections"],
    })


@app.route("/api/session", methods=["GET"])
def api_get_session():
    if not _initialize_runtime():
        return jsonify({"error": _runtime_error or "Detector runtime is unavailable."}), 503

    report_info = detector.get_report_info()
    return jsonify({
        "session_active": detector.is_enabled(),
        "status_message": detector.get_status_message(),
        "report_available": report_info["available"],
        "report_filename": report_info["filename"],
        "report_summary": report_info["summary"],
        "report_total_detections": report_info["total_detections"],
    })


@app.route("/api/session", methods=["POST"])
def api_toggle_session():
    if not _initialize_runtime():
        return jsonify({"error": _runtime_error or "Detector runtime is unavailable."}), 503

    data = request.get_json(silent=True) or {}
    action = data.get("action", "")

    if action == "stop":
        detector.set_enabled(False)
    elif action == "start":
        detector.set_enabled(True)
    else:
        detector.set_enabled(not detector.is_enabled())

    report_info = detector.get_report_info()
    return jsonify({
        "session_active": detector.is_enabled(),
        "status_message": detector.get_status_message(),
        "report_available": report_info["available"],
        "report_filename": report_info["filename"],
        "report_summary": report_info["summary"],
        "report_total_detections": report_info["total_detections"],
    })


@app.route("/api/session/report")
def api_session_report():
    if not _initialize_runtime():
        return jsonify({"error": _runtime_error or "Detector runtime is unavailable."}), 503

    report = detector.get_report_payload()
    if not report:
        return jsonify({"available": False, "filename": None, "summary": [], "total_detections": 0})

    return jsonify({
        "available": True,
        "filename": report["filename"],
        "summary": report["summary"],
        "total_detections": report["total_detections"],
        "generated_at": report["ended_at"],
    })


@app.route("/api/session/report/download")
def api_download_session_report():
    if not _initialize_runtime():
        return jsonify({"error": _runtime_error or "Detector runtime is unavailable."}), 503

    report = detector.get_report_payload()
    if not report:
        return jsonify({"error": "No report is available yet."}), 404

    response = make_response(report["pdf_content"])
    response.headers["Content-Disposition"] = f"attachment; filename={report['filename']}"
    response.mimetype = "application/pdf"
    return response


if __name__ == "__main__":
    try:
        # threaded=True lets Flask handle the long-lived /video_feed
        # connection concurrently with short-lived /api/* polling requests.
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    finally:
        if detector is not None:
            detector.stop()
        if camera is not None:
            camera.stop()
