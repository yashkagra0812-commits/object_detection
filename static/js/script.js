/**
 * script.js - Frontend logic for the real-time object detection UI.
 */

const POLL_INTERVAL_MS = 500;
let sessionActive = true;

const els = {
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  cameraFps: document.getElementById("cameraFps"),
  inferenceFps: document.getElementById("inferenceFps"),
  latency: document.getElementById("latency"),
  objectCount: document.getElementById("objectCount"),
  detectionsList: document.getElementById("detectionsList"),
  videoStream: document.getElementById("videoStream"),
  ipCameraUrl: document.getElementById("ipCameraUrl"),
  connectIpCamera: document.getElementById("connectIpCamera"),
  ipCameraMessage: document.getElementById("ipCameraMessage"),
  currentSourceLabel: document.getElementById("currentSourceLabel"),
  sessionToggle: document.getElementById("sessionToggle"),
  sessionSummary: document.getElementById("sessionSummary"),
  sessionPill: document.getElementById("sessionPill"),
  downloadReportBtn: document.getElementById("downloadReportBtn"),
};

function setIpCameraMessage(message, status = "info") {
  if (els.ipCameraMessage) {
    els.ipCameraMessage.textContent = message;
    els.ipCameraMessage.className = `connection-message ${status}`;
  }
}

function validateIpCameraUrl(url) {
  if (!url || !url.trim()) {
    return { valid: false, message: "URL cannot be blank." };
  }

  const trimmed = url.trim();
  const lower = trimmed.toLowerCase();
  const accepted = ["http://", "https://", "rtsp://"];
  if (!accepted.some((prefix) => lower.startsWith(prefix))) {
    return { valid: false, message: "Please enter a valid IP webcam URL." };
  }

  return { valid: true, message: "" };
}

function resetToBackendStream(message = "Connection failed. Reverting to backend camera.") {
  if (els.currentSourceLabel) {
    els.currentSourceLabel.textContent = "Current source: Backend camera";
  }
  setIpCameraMessage(message, "error");
  if (els.videoStream) {
    els.videoStream.src = `/video_feed?retry=${Date.now()}`;
  }
}

async function connectIpCameraStream(url) {
  const validation = validateIpCameraUrl(url);
  if (!validation.valid) {
    setIpCameraMessage(validation.message, "error");
    return;
  }

  const normalized = url.trim();
  if (els.connectIpCamera) {
    els.connectIpCamera.disabled = true;
  }
  setIpCameraMessage("Connecting...", "loading");

  try {
    const res = await fetch("/api/connect_camera", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: normalized }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }

    if (els.currentSourceLabel) {
      els.currentSourceLabel.textContent = `Current source: ${data.url}`;
    }
    localStorage.setItem("ipCameraUrl", data.url);
    setIpCameraMessage("Connected", "success");
    refreshVideoFeed();
  } catch (error) {
    resetToBackendStream(`Connection failed. ${error.message}`);
  } finally {
    if (els.connectIpCamera) {
      els.connectIpCamera.disabled = false;
    }
  }
}

function refreshVideoFeed() {
  if (els.videoStream) {
    els.videoStream.src = `/video_feed?retry=${Date.now()}`;
  }
}

function loadStoredIpCameraUrl() {
  const storedUrl = localStorage.getItem("ipCameraUrl");
  if (storedUrl && els.ipCameraUrl) {
    els.ipCameraUrl.value = storedUrl;
  }
}

function updateSessionUI(state) {
  sessionActive = Boolean(state.session_active);
  if (els.sessionToggle) {
    els.sessionToggle.textContent = sessionActive ? "Stop Session" : "Start Session";
  }
  if (els.sessionSummary) {
    els.sessionSummary.textContent = state.status_message || "Session status unavailable.";
  }
  if (els.sessionPill) {
    els.sessionPill.textContent = sessionActive ? "Running" : "Paused";
    els.sessionPill.classList.toggle("paused", !sessionActive);
  }
  if (els.downloadReportBtn) {
    const available = Boolean(state.report_available);
    els.downloadReportBtn.disabled = !available;
    els.downloadReportBtn.classList.toggle("is-ready", available);
  }
}

async function pollStats() {
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (els.cameraFps) els.cameraFps.textContent = data.camera_fps;
    if (els.inferenceFps) els.inferenceFps.textContent = data.inference_fps;
    if (els.latency) els.latency.textContent = `${data.inference_ms} ms`;
    if (els.objectCount) els.objectCount.textContent = data.object_count;

    updateSessionUI(data);
    setConnected(true);
  } catch (err) {
    setConnected(false);
  }
}

async function pollDetections() {
  try {
    const res = await fetch("/api/detections");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const detections = await res.json();
    renderDetections(detections, sessionActive);
  } catch (err) {
    // Connection status is already reflected by pollStats(); nothing further to do here.
  }
}

function renderDetections(detections, activeSession = true) {
  if (els.detectionsList) {
    els.detectionsList.innerHTML = "";
  }

  if (!activeSession) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Session paused. Start it to resume detection.";
    els.detectionsList.appendChild(li);
    return;
  }

  if (!detections.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No objects detected yet.";
    els.detectionsList.appendChild(li);
    return;
  }

  detections
    .slice()
    .sort((a, b) => b.confidence - a.confidence)
    .forEach((det) => {
      const li = document.createElement("li");
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = det.label;
      const confidence = document.createElement("span");
      confidence.className = "confidence";
      confidence.textContent = `${(det.confidence * 100).toFixed(0)}%`;
      li.appendChild(label);
      li.appendChild(confidence);
      els.detectionsList.appendChild(li);
    });
}

function setConnected(isConnected) {
  if (els.statusDot) {
    els.statusDot.classList.toggle("online", isConnected);
    els.statusDot.classList.toggle("offline", !isConnected);
  }
  if (els.statusText) {
    els.statusText.textContent = isConnected ? (sessionActive ? "Live" : "Paused") : "Reconnecting…";
  }
}

async function toggleSession() {
  const action = sessionActive ? "stop" : "start";
  try {
    const res = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const state = await res.json();
    updateSessionUI(state);
    setConnected(true);
  } catch (error) {
    setConnected(false);
  }
}

async function downloadReport() {
  if (!els.downloadReportBtn || els.downloadReportBtn.disabled) {
    return;
  }

  try {
    const res = await fetch("/api/session/report/download");
    if (!res.ok) throw new Error("No report is available yet.");

    const contentDisposition = res.headers.get("Content-Disposition") || "";
    const fileMatch = contentDisposition.match(/filename="?([^";]+)"?/i);
    const filename = fileMatch ? fileMatch[1] : "object-detection-report.csv";
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error(error);
  }
}

if (els.videoStream) {
  els.videoStream.addEventListener("error", () => {
    setTimeout(() => {
      refreshVideoFeed();
    }, 1000);
  });
}

if (els.connectIpCamera) {
  els.connectIpCamera.addEventListener("click", () => {
    connectIpCameraStream(els.ipCameraUrl.value);
  });
}

if (els.sessionToggle) {
  els.sessionToggle.addEventListener("click", toggleSession);
}

if (els.downloadReportBtn) {
  els.downloadReportBtn.addEventListener("click", downloadReport);
}

loadStoredIpCameraUrl();
setInterval(pollStats, POLL_INTERVAL_MS);
setInterval(pollDetections, POLL_INTERVAL_MS);
pollStats();
pollDetections();
