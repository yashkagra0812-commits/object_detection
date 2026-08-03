"""
Central configuration for the real-time object detection app.

Tweak these values to tune performance vs. accuracy for your hardware.
"""

import os


# ---- Camera settings ----
# Use an integer for a local camera index, or a string URL for an IP webcam.
# Examples:
#   CAMERA_SOURCE = 0
#   CAMERA_SOURCE = "http://192.168.1.100:8080/video"
# You can also set the environment variable CAMERA_SOURCE before launching
# the app, for example: $env:CAMERA_SOURCE="http://192.168.1.100:8080/video"

def _resolve_camera_source():
    value = os.getenv("CAMERA_SOURCE", "0").strip()
    if value.isdigit():
        return int(value)
    return value


CAMERA_SOURCE = _resolve_camera_source()
FRAME_WIDTH = 640          # lower resolution = higher FPS
FRAME_HEIGHT = 480
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# ---- Model settings ----
# The default model now uses YOLOv8 small for a stronger balance of speed
# and accuracy. You can override it with MODEL_NAME or the environment
# variable MODEL_NAME if you want a heavier model on a stronger machine.
MODEL_NAME = os.getenv("MODEL_NAME", "yolov8n.onnx")
CONFIDENCE_THRESHOLD = 0.35   # lower threshold helps catch more objects
IOU_THRESHOLD = 0.45          # non-max suppression overlap threshold
DEVICE = "auto"               # "cpu", "cuda", or "auto" (auto-detects a GPU)

# ---- Streaming settings ----
JPEG_QUALITY = 85             # 0-100, lower = faster encode, smaller payload
TARGET_STREAM_FPS = 24        # smooth live preview without overloading the CPU
