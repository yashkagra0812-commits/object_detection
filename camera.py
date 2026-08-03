"""
camera.py - Threaded camera capture.

Why a dedicated thread?
------------------------
cv2.VideoCapture.read() blocks until a new frame is ready. If whatever is
consuming frames (JPEG encoding, model inference, etc.) is slower than the
camera's native frame rate, frames pile up in an internal buffer. The result
is the classic "laggy webcam" effect, where the video you see is several
frames behind real time and gets progressively more delayed.

CameraStream solves this with a producer/consumer pattern:
  - A background thread continuously calls cap.read() as fast as the camera
    allows and immediately overwrites `self.frame` with the newest image.
  - Consumers call `.read()` and always get the *latest* available frame,
    never a stale, queued-up one. Old frames that were never consumed are
    simply discarded, which is exactly the right trade-off for a live
    preview (we care about "now", not "every frame that ever existed").

This class also tracks a rolling FPS counter so the frontend can display
real camera throughput.
"""
import os
import threading
import time

import cv2

import config


class CameraStream:
    def __init__(self, src=config.CAMERA_SOURCE):
        self.cap_lock = threading.Lock()
        self.source = src
        self.cap = self._create_capture(src)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source {src}. "
                "If using a local webcam, check that it is connected and not "
                "already in use. If using an IP webcam, verify the URL and "
                "network connectivity."
            )

        self.lock = threading.Lock()
        self.frame = None
        self.fps = 0.0
        self._frame_times = []

        self.running = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def _create_capture(self, src):
        backends = []
        if isinstance(src, int) and os.name == "nt":
            backends.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF])
        backends.append(-1)

        for backend in backends:
            if backend == -1:
                cap = cv2.VideoCapture(src)
            else:
                cap = cv2.VideoCapture(src, backend)

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                return cap
            cap.release()

        return cv2.VideoCapture(src)

    def start(self):
        self.running = True
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            with self.cap_lock:
                cap = self.cap
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            now = time.time()
            with self.lock:
                self.frame = frame
                self._frame_times.append(now)
                cutoff = now - 1.0
                self._frame_times = [t for t in self._frame_times if t >= cutoff]
                self.fps = float(len(self._frame_times))

    def read(self):
        """Return (latest_frame_copy, current_fps). Frame is None until the
        camera has produced its first frame."""
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.fps

    def switch_source(self, src):
        new_cap = self._create_capture(src)

        if not new_cap.isOpened():
            new_cap.release()
            raise RuntimeError(
                f"Could not open camera source {src}. "
                "If using a local webcam, verify it is connected and not "
                "already in use. If using an IP webcam, verify the URL and "
                "network connectivity."
            )

        with self.cap_lock:
            old_cap = self.cap
            self.cap = new_cap
            self.source = src
            with self.lock:
                self.frame = None
                self.fps = 0.0

        old_cap.release()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        with self.cap_lock:
            self.cap.release()
