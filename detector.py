"""
detector.py - Object detection worker.

Runs a YOLOv8 model in its own background thread, completely decoupled from
camera capture and video streaming so the live preview stays smooth.
"""
import json
import os
import threading
import time
from collections import Counter
from datetime import datetime, timezone

import cv2
import numpy as np
import onnxruntime as ort

import config

COCO_CLASS_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

_rng = np.random.default_rng(42)  # fixed seed -> stable per-class colors


class Detection:
    """A single detected object."""

    __slots__ = ("cls_id", "label", "confidence", "box")

    def __init__(self, cls_id, label, confidence, box):
        self.cls_id = cls_id
        self.label = label
        self.confidence = confidence
        self.box = box  # (x1, y1, x2, y2) in pixel coordinates


class DetectorThread:
    def __init__(self, camera_stream):
        self.camera_stream = camera_stream
        self.device = self._resolve_device()
        self.model = self._load_model()

        self.class_names = self.model.names  # {cls_id: label}
        self.colors = {
            cls_id: tuple(int(c) for c in _rng.integers(60, 255, size=3))
            for cls_id in self.class_names
        }

        self.lock = threading.Lock()
        self.detections = []
        self.inference_fps = 0.0
        self.last_inference_ms = 0.0
        self.session_active = True
        self.status_message = "Session running"

        self.session_events = []
        self.session_summary = Counter()
        self.session_started_at = None
        self.session_report_payload = None

        self.running = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _resolve_device(self):
        if config.DEVICE.lower() == "cuda":
            return "cuda"
        return "cpu"

    def _load_model(self):
        model_path = config.MODEL_NAME
        if not model_path.lower().endswith(".onnx"):
            raise RuntimeError(
                "MODEL_NAME must point to an ONNX model when using the ONNX runtime pipeline."
            )
        if not os.path.exists(model_path):
            raise RuntimeError(f"ONNX model file not found: {model_path}")

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            return ort.InferenceSession(model_path, providers=providers)
        except Exception as exc:
            raise RuntimeError(f"Unable to load ONNX model: {exc}")

    def _letterbox(self, image, new_shape=(640, 640), color=(114, 114, 114)):
        shape = image.shape[:2]  # height, width
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, r, (dw, dh)

    def _xywh2xyxy(self, x):
        y = x.copy()
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def _scale_coords(self, boxes, ratio, dwdh):
        boxes[:, [0, 2]] -= dwdh[0]
        boxes[:, [1, 3]] -= dwdh[1]
        boxes /= ratio
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, config.FRAME_WIDTH)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, config.FRAME_HEIGHT)
        return boxes.round().astype(np.int32)

    def _postprocess(self, output):
        if output is None or len(output.shape) != 3:
            return []

        output = output[0]
        if output.size == 0:
            return []

        boxes = output[:, :4]
        scores = output[:, 4:5] * output[:, 5:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= config.CONFIDENCE_THRESHOLD
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        boxes = self._xywh2xyxy(boxes)
        boxes = self._scale_coords(boxes, self.ratio, self.dwdh)

        xywh = boxes.copy()
        xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
        xywh[:, 3] = xywh[:, 3] - xywh[:, 1]

        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            confidences.tolist(),
            float(config.CONFIDENCE_THRESHOLD),
            float(config.IOU_THRESHOLD),
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten() if isinstance(indices, (list, tuple, np.ndarray)) else np.array([indices]).flatten()
        detections = []
        for idx in indices:
            cls_id = int(class_ids[idx])
            detections.append(
                Detection(
                    cls_id,
                    COCO_CLASS_NAMES[cls_id] if cls_id < len(COCO_CLASS_NAMES) else f"class_{cls_id}",
                    float(confidences[idx]),
                    tuple(boxes[idx].tolist()),
                )
            )
        return detections

    def start(self):
        self.running = True
        self.thread.start()
        return self

    def is_enabled(self):
        with self.lock:
            return self.session_active

    def get_status_message(self):
        with self.lock:
            return self.status_message

    def _format_timestamp(self, timestamp):
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    def _start_new_session_locked(self):
        self.session_events = []
        self.session_summary = Counter()
        self.session_started_at = time.time()
        self.session_report_payload = None

    def _escape_pdf_text(self, text):
        return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def _build_pdf_bytes(self, payload):
        lines = [
            "Object Detection Session Report",
            "",
            f"Started: {payload['started_at'] or 'N/A'}",
            f"Ended: {payload['ended_at'] or 'N/A'}",
            f"Total detections: {payload['total_detections']}",
            "",
            "Summary:",
        ]
        for item in payload["summary"]:
            lines.append(f"- {item['label']}: {item['count']}")

        lines.extend(["", "Detected events:"])
        if payload["events"]:
            for event in payload["events"]:
                lines.append(
                    f"{event['timestamp']} | {event['label']} | conf={event['confidence']} | box={json.dumps(event['box'])}"
                )
        else:
            lines.append("No detections were recorded during this session.")

        content_lines = ["BT", "/F1 12 Tf"]
        y_position = 760
        for index, line in enumerate(lines):
            if index == 0:
                content_lines.append(f"72 {y_position} Td")
            else:
                content_lines.append("0 -14 Td")
            content_lines.append(f"({self._escape_pdf_text(line)}) Tj")

        content_lines.append("ET")
        content_stream = "\n".join(content_lines)
        stream_bytes = content_stream.encode("latin-1", "replace")

        objects = []
        objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
        objects.append(
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
            b"endobj\n"
        )
        objects.append(
            b"4 0 obj\n"
            b"<< /Length 0 >>\n"
            b"stream\n"
        )
        objects.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

        content_obj = bytearray()
        content_obj.extend(b"4 0 obj\n<< /Length ")
        content_obj.extend(str(len(stream_bytes)).encode("ascii"))
        content_obj.extend(b" >>\nstream\n")
        content_obj.extend(stream_bytes)
        content_obj.extend(b"\nendstream\nendobj\n")

        pdf_parts = [b"%PDF-1.4\n"]
        offsets = [0]
        for obj in objects[:-1]:
            offsets.append(len(b"".join(pdf_parts)))
            pdf_parts.append(obj)
        offsets.append(len(b"".join(pdf_parts)))
        pdf_parts.append(content_obj)

        pdf_bytes = b"".join(pdf_parts)
        xref_offset = len(pdf_bytes)
        xref_lines = [b"xref\n0 6\n0000000000 65535 f \n"]
        for offset in offsets[1:]:
            xref_lines.append(f"{offset:010d} 00000 n \n".encode("ascii"))
        xref_bytes = b"".join(xref_lines)
        trailer = (
            b"trailer\n"
            b"<< /Size 6 /Root 1 0 R >>\n"
            b"startxref\n"
            + str(xref_offset).encode("ascii") + b"\n"
            + b"%%EOF"
        )
        return pdf_bytes + xref_bytes + trailer

    def _finalize_session_locked(self):
        started_at = self.session_started_at or time.time()
        ended_at = time.time()
        events = list(self.session_events)
        summary = [
            {"label": label, "count": count}
            for label, count in sorted(self.session_summary.items())
        ]

        filename = f"object-detection-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
        payload = {
            "filename": filename,
            "pdf_content": self._build_pdf_bytes({
                "started_at": self._format_timestamp(started_at),
                "ended_at": self._format_timestamp(ended_at),
                "total_detections": len(events),
                "summary": summary,
                "events": events,
            }),
            "started_at": self._format_timestamp(started_at),
            "ended_at": self._format_timestamp(ended_at),
            "total_detections": len(events),
            "summary": summary,
            "events": events,
        }
        self.session_report_payload = payload
        return payload

    def set_enabled(self, enabled):
        with self.lock:
            new_state = bool(enabled)
            if new_state == self.session_active:
                self.status_message = "Session running" if self.session_active else "Session paused"
                return

            self.session_active = new_state
            if self.session_active:
                self.status_message = "Session running"
                self._start_new_session_locked()
            else:
                self.status_message = "Session paused"
                self._finalize_session_locked()
                self.detections = []
                self.inference_fps = 0.0
                self.last_inference_ms = 0.0

    def _run(self):
        frame_times = []
        while self.running:
            if not self.is_enabled():
                time.sleep(0.1)
                continue

            frame, _ = self.camera_stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            img, self.ratio, self.dwdh = self._letterbox(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                new_shape=(config.INPUT_HEIGHT, config.INPUT_WIDTH),
            )
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[None]

            outputs = self.model.run(None, {self.model.get_inputs()[0].name: img})
            detections = self._postprocess(outputs[0])
            elapsed_ms = (time.time() - t0) * 1000

            now = time.time()
            with self.lock:
                self.detections = detections
                self.last_inference_ms = elapsed_ms
                frame_times.append(now)
                cutoff = now - 1.0
                frame_times = [t for t in frame_times if t >= cutoff]
                self.inference_fps = float(len(frame_times))
                if self.session_active:
                    if self.session_started_at is None:
                        self.session_started_at = time.time()
                    for det in detections:
                        self.session_events.append(
                            {
                                "timestamp": self._format_timestamp(now),
                                "label": det.label,
                                "confidence": round(float(det.confidence), 3),
                                "box": list(det.box),
                            }
                        )
                        self.session_summary[det.label] += 1

    def get_detections(self):
        """Return (detections_list, inference_fps, last_inference_ms, status_message)."""
        with self.lock:
            return list(self.detections), self.inference_fps, self.last_inference_ms, self.status_message

    def get_report_payload(self):
        with self.lock:
            return self.session_report_payload

    def get_report_info(self):
        with self.lock:
            if not self.session_report_payload:
                return {"available": False, "filename": None, "summary": [], "total_detections": 0}
            return {
                "available": True,
                "filename": self.session_report_payload["filename"],
                "summary": self.session_report_payload["summary"],
                "total_detections": self.session_report_payload["total_detections"],
                "generated_at": self.session_report_payload["ended_at"],
            }

    def draw(self, frame):
        """Draw the most recent detections onto `frame` in place and return it."""
        detections, _, _, status_message = self.get_detections()
        if not self.is_enabled():
            cv2.putText(
                frame,
                status_message,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press Start to resume detection",
                (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            return frame

        for det in detections:
            x1, y1, x2, y2 = det.box
            color = self.colors.get(det.cls_id, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = f"{det.label} {det.confidence * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1, th + 8)
            cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 4, label_y), color, -1)
            cv2.putText(
                frame,
                label_text,
                (x1 + 2, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
