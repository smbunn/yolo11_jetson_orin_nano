#!/usr/bin/env python3
"""
YOLO11 Model Comparison App for Jetson Orin Nano
Compares Detection, Segmentation, and Pose models across all sizes (n/s/m/l/x)
with support for PyTorch (.pt) and TensorRT (.engine) inference.

Requirements:
    pip install ultralytics opencv-python-headless PyQt5 numpy

TensorRT engine export (run once per model):
    from ultralytics import YOLO
    YOLO("yolo11n.pt").export(format="engine", half=True, device=0)
    YOLO("yolo11n-seg.pt").export(format="engine", half=True, device=0)
    YOLO("yolo11n-pose.pt").export(format="engine", half=True, device=0)
"""

import sys
import os

# ── Jetson fixes ───────────────────────────────────────────────────────────────
# 1. Remove OpenCV's bundled Qt plugin path so it doesn't conflict with PyQt5
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

# 2. Suppress ONNX Runtime's GPU device-discovery warning on Jetson
#    (Jetson has a unified GPU, not a discrete PCIe card — the warning is harmless)
os.environ["ORT_DISABLE_EXCEPTIONS"] = "1"
os.environ.setdefault("ONNXRUNTIME_PROVIDERS", "CPUExecutionProvider")

# 3. Fix numpy conflict: make sure the system numpy is found before cv2 loads
#    its own bundled copy.
import numpy as np   # import numpy FIRST before cv2
# ───────────────────────────────────────────────────────────────────────────────

import time
import threading
import queue
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider, QGroupBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QDialog, QDialogButtonBox, QRadioButton, QButtonGroup, QListWidget,
    QListWidgetItem, QCheckBox, QProgressBar, QFrame, QSpinBox,
    QDoubleSpinBox, QTextEdit, QScrollArea, QSizePolicy, QMessageBox
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QMutex, QMutexLocker
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

WEIGHTS_DIR = Path("./weights")       # Where .pt / .engine files live
MODEL_SIZES = ["n", "s", "m", "l", "x"]
TRT_PRECISIONS = ["fp16", "fp32", "int8"]  # checked in preference order

TASK_CONFIGS = {
    "Detection": {
        "pt_pattern":  "yolo11{size}.pt",
        "trt_pattern": "yolo11{size}{prec_suffix}.engine",
        "color":       "#E8622A",
    },
    "Segmentation": {
        "pt_pattern":  "yolo11{size}-seg.pt",
        "trt_pattern": "yolo11{size}-seg{prec_suffix}.engine",
        "color":       "#2A8CE8",
    },
    "Pose": {
        "pt_pattern":  "yolo11{size}-pose.pt",
        "trt_pattern": "yolo11{size}-pose{prec_suffix}.engine",
        "color":       "#2AE862",
    },
}

DEVICE_OPTIONS = ["cpu", "0"]   # cpu or CUDA/TensorRT on GPU 0


# ─────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────

@dataclass
class ModelInfo:
    task:      str
    size:      str
    backend:   str        # "PT" or "TRT"
    path:      Path
    precision: str = ""   # "fp16" / "fp32" / "int8" for TRT, "" for PT
    loaded:    bool = False
    model:     object = None

    @property
    def name(self) -> str:
        suffix = "-seg" if self.task == "Segmentation" \
                 else ("-pose" if self.task == "Pose" else "")
        return f"yolo11{self.size}{suffix}"

    @property
    def label(self) -> str:
        if self.backend == "TRT" and self.precision:
            return f"{self.name} [TRT-{self.precision.upper()}]"
        return f"{self.name} [{self.backend}]"


@dataclass
class BenchResult:
    model_label: str
    avg_fps:     float = 0.0
    avg_ms:      float = 0.0
    min_ms:      float = 0.0
    max_ms:      float = 0.0
    n_objects:   float = 0.0
    frames:      int   = 0


# ─────────────────────────────────────────────
#  Model discovery
# ─────────────────────────────────────────────

def discover_models() -> list[ModelInfo]:
    """
    Scan WEIGHTS_DIR (and cwd) for available model files.

    Recognised TRT engine naming conventions:
      yolo11n-fp16.engine   (precision-tagged -- preferred)
      yolo11n.engine        (legacy untagged -- treated as fp16)
    """
    search_dirs = [WEIGHTS_DIR, Path(".")]
    found = []

    for task, cfg in TASK_CONFIGS.items():
        for size in MODEL_SIZES:

            # ── PT weights ──────────────────────
            pt_name = cfg["pt_pattern"].format(size=size)
            for d in search_dirs:
                fp = d / pt_name
                if fp.exists():
                    found.append(ModelInfo(task=task, size=size,
                                           backend="PT", path=fp,
                                           precision=""))
                    break

            # ── TRT engines (precision-tagged) ──
            for prec in TRT_PRECISIONS:
                trt_name = cfg["trt_pattern"].format(size=size,
                                                     prec_suffix=f"-{prec}")
                for d in search_dirs:
                    fp = d / trt_name
                    if fp.exists():
                        found.append(ModelInfo(task=task, size=size,
                                               backend="TRT", path=fp,
                                               precision=prec))
                        break

            # ── TRT engine (legacy untagged) ────
            legacy_name = cfg["trt_pattern"].format(size=size, prec_suffix="")
            for d in search_dirs:
                fp = d / legacy_name
                if fp.exists():
                    # Only add if no fp16 engine already found (legacy ≈ fp16)
                    already = any(
                        m.task == task and m.size == size
                        and m.backend == "TRT" and m.precision == "fp16"
                        for m in found
                    )
                    if not already:
                        found.append(ModelInfo(task=task, size=size,
                                               backend="TRT", path=fp,
                                               precision="fp16"))
                    break

    return found


# ─────────────────────────────────────────────
#  Shared camera thread (one per source)
#  Captures frames and fans them out to
#  multiple subscribers via callbacks.
# ─────────────────────────────────────────────

class CameraThread(QThread):
    """Opens the camera once and delivers copies to all registered listeners.
    Automatically stops and releases the device when the last listener leaves."""

    def __init__(self, source):
        super().__init__()
        self.source      = source
        self._running    = False
        self._lock       = threading.Lock()
        self._listeners  = []

    @property
    def listener_count(self):
        with self._lock:
            return len(self._listeners)

    def add_listener(self, fn):
        with self._lock:
            self._listeners.append(fn)

    def remove_listener(self, fn):
        with self._lock:
            self._listeners = [l for l in self._listeners if l is not fn]
            remaining = len(self._listeners)
        # Stop the camera thread when nobody is listening any more
        if remaining == 0:
            self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            return

        self._running = True
        while self._running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            with self._lock:
                listeners = list(self._listeners)
            for fn in listeners:
                try:
                    fn(frame.copy())
                except Exception:
                    pass

        cap.release()
        # Remove from registry so next use creates a fresh thread
        key = str(self.source)
        with _camera_registry_lock:
            if _camera_registry.get(key) is self:
                _camera_registry.pop(key, None)

    def stop(self):
        self._running = False
        self.wait(3000)


# Registry so multiple InferenceThreads sharing the same source
# reuse a single CameraThread.
_camera_registry: dict = {}
_camera_registry_lock = threading.Lock()

def get_shared_camera(source) -> CameraThread:
    key = str(source)
    with _camera_registry_lock:
        if key not in _camera_registry or not _camera_registry[key].isRunning():
            cam = CameraThread(source)
            _camera_registry[key] = cam
            cam.start()
        return _camera_registry[key]

def release_shared_camera(source):
    key = str(source)
    with _camera_registry_lock:
        cam = _camera_registry.pop(key, None)
    if cam:
        cam.stop()

def release_all_cameras():
    """Stop every active CameraThread and clear the registry.
    Call this before any code that needs exclusive camera access (e.g. Benchmark)."""
    with _camera_registry_lock:
        cams = list(_camera_registry.values())
        _camera_registry.clear()
    for cam in cams:
        cam.stop()   # blocks until the thread exits and cap.release() is called


# Map our task names to ultralytics task strings.
# Essential for .engine files which have no embedded task metadata.
_TASK_MAP = {
    "Detection":    "detect",
    "Segmentation": "segment",
    "Pose":         "pose",
}

def load_yolo(model_info):
    """Load a YOLO model with explicit task so .engine files work correctly."""
    from ultralytics import YOLO
    task = _TASK_MAP.get(model_info.task, "detect")
    return YOLO(str(model_info.path), task=task)


# ─────────────────────────────────────────────
#  Inference worker thread
# ─────────────────────────────────────────────

class InferenceThread(QThread):
    frame_ready   = pyqtSignal(np.ndarray, dict)   # annotated frame + stats
    error_signal  = pyqtSignal(str)

    def __init__(self, model_info: ModelInfo, source, confidence: float = 0.5,
                 device: str = "0"):
        super().__init__()
        self.model_info = model_info
        self.source     = source
        self.confidence = confidence
        self.device     = device
        self._running   = False
        self._mutex     = QMutex()
        self._frame_queue = queue.Queue(maxsize=2)

    def _on_camera_frame(self, frame: np.ndarray):
        """Called by CameraThread — drop frames if inference is behind."""
        if not self._frame_queue.full():
            self._frame_queue.put(frame)

    def run(self):
        try:
            model = load_yolo(self.model_info)
            self.model_info.model  = model
            self.model_info.loaded = True
        except Exception as e:
            self.error_signal.emit(f"Failed to load {self.model_info.label}: {e}")
            return

        # Use shared camera if source is a device, else open privately (for files)
        use_shared = (str(self.source).startswith("/dev/") or
                      str(self.source).isdigit() or
                      isinstance(self.source, int))

        if use_shared:
            cam = get_shared_camera(self.source)
            cam.add_listener(self._on_camera_frame)
        else:
            # Video file — open privately
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.error_signal.emit(f"Cannot open source: {self.source}")
                return

        self._running = True
        frame_times   = []

        while self._running:
            if use_shared:
                try:
                    frame = self._frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break

            t0 = time.perf_counter()
            try:
                results = model.predict(
                    frame,
                    conf=self.confidence,
                    device=self.device,
                    verbose=False
                )
            except Exception as e:
                self.error_signal.emit(str(e))
                break
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000
            frame_times.append(elapsed_ms)
            if len(frame_times) > 30:
                frame_times.pop(0)

            annotated = results[0].plot()
            n_objects = len(results[0].boxes) if results[0].boxes is not None else 0

            stats = {
                "fps":      1000 / elapsed_ms if elapsed_ms > 0 else 0,
                "ms":       elapsed_ms,
                "avg_fps":  1000 / np.mean(frame_times),
                "objects":  n_objects,
                "model":    self.model_info.label,
                "task":     self.model_info.task,
            }
            self.frame_ready.emit(annotated, stats)

        # Cleanup
        if use_shared:
            # Remove our listener — CameraThread auto-stops when count hits 0
            key = str(self.source)
            with _camera_registry_lock:
                cam = _camera_registry.get(key)
            if cam:
                cam.remove_listener(self._on_camera_frame)
        else:
            cap.release()

    def stop(self):
        self._running = False
        self.wait(3000)


# ─────────────────────────────────────────────
#  Benchmark thread
# ─────────────────────────────────────────────

class BenchmarkThread(QThread):
    progress   = pyqtSignal(int, str)         # percent, message
    result     = pyqtSignal(BenchResult)
    finished_all = pyqtSignal(list)

    def __init__(self, models: list[ModelInfo], source, confidence: float,
                 device: str, n_frames: int = 100):
        super().__init__()
        self.models     = models
        self.source     = source
        self.confidence = confidence
        self.device     = device
        self.n_frames   = n_frames
        self._abort     = False

    def run(self):
        results = []
        total = len(self.models)

        # Ensure no other tab is holding the camera open before we try to grab it.
        release_all_cameras()

        # Open camera once and reuse across all models.
        # This avoids the /dev/video0 exclusive-access problem and
        # gives a clear error if the source can't be opened at all.
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.progress.emit(0, f"ERROR: Cannot open source: {self.source}")
            self.finished_all.emit([])
            return

        # Grab a pool of frames upfront so we benchmark inference,
        # not camera I/O, and avoid re-opening the device per model.
        self.progress.emit(0, "Capturing frames for benchmark…")
        frame_pool = []
        pool_size  = min(self.n_frames + 10, 120)
        while len(frame_pool) < pool_size:
            ret, frame = cap.read()
            if not ret:
                break
            frame_pool.append(frame)
        cap.release()

        if not frame_pool:
            self.progress.emit(0, f"ERROR: Could not read frames from: {self.source}")
            self.finished_all.emit([])
            return

        for idx, mi in enumerate(self.models):
            if self._abort:
                break
            self.progress.emit(
                int(idx / total * 100),
                f"Loading {mi.label}…"
            )
            try:
                model = load_yolo(mi)
            except Exception as e:
                self.progress.emit(int(idx / total * 100),
                                   f"Error loading {mi.label}: {e}")
                continue

            frame_ms  = []
            obj_counts = []

            # Warm-up pass (not timed)
            try:
                model.predict(frame_pool[0], conf=self.confidence,
                              device=self.device, verbose=False)
            except Exception:
                pass

            for f in range(self.n_frames):
                if self._abort:
                    break
                frame = frame_pool[f % len(frame_pool)]

                t0 = time.perf_counter()
                try:
                    res = model.predict(frame, conf=self.confidence,
                                        device=self.device, verbose=False)
                except Exception as e:
                    self.progress.emit(int(idx / total * 100),
                                       f"Inference error on {mi.label}: {e}")
                    break
                t1 = time.perf_counter()

                ms = (t1 - t0) * 1000
                frame_ms.append(ms)

                # Count detected objects — handle det/seg/pose results
                r = res[0]
                if r.boxes is not None:
                    n = len(r.boxes)
                elif r.masks is not None:
                    n = len(r.masks)
                elif r.keypoints is not None:
                    n = len(r.keypoints)
                else:
                    n = 0
                obj_counts.append(n)

                self.progress.emit(
                    int((idx + (f + 1) / self.n_frames) / total * 100),
                    f"Benchmarking {mi.label} — frame {f+1}/{self.n_frames}"
                )

            if frame_ms:
                br = BenchResult(
                    model_label=mi.label,
                    avg_fps=1000 / np.mean(frame_ms),
                    avg_ms=np.mean(frame_ms),
                    min_ms=np.min(frame_ms),
                    max_ms=np.max(frame_ms),
                    n_objects=np.mean(obj_counts),
                    frames=len(frame_ms),
                )
                results.append(br)
                self.result.emit(br)

        self.progress.emit(100, f"Done — {len(results)} models benchmarked")
        self.finished_all.emit(results)

    def abort(self):
        self._abort = True


# ─────────────────────────────────────────────
#  Video display widget
# ─────────────────────────────────────────────

class VideoLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background:#111;border:1px solid #333;")
        self.setText("No feed")
        self.setFont(QFont("Monospace", 10))
        self.setStyleSheet(
            "background:#0a0a0a; color:#555; border:1px solid #2a2a2a;"
            "border-radius:4px; font-family:monospace;"
        )

    def update_frame(self, frame: np.ndarray):
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w  = rgb.shape[:2]
        img   = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix   = QPixmap.fromImage(img)
        self.setPixmap(pix.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))


# ─────────────────────────────────────────────
#  Per-model panel (used in Compare tab)
# ─────────────────────────────────────────────

class ModelPanel(QWidget):
    def __init__(self, model_info: ModelInfo, source, confidence: float,
                 device: str, parent=None):
        super().__init__(parent)
        self.model_info = model_info
        self.source     = source
        self.confidence = confidence
        self.device     = device
        self.thread: Optional[InferenceThread] = None
        self._build_ui()
        self._start()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        color = TASK_CONFIGS[self.model_info.task]["color"]
        title = QLabel(self.model_info.label)
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Monospace", 9, QFont.Bold))
        title.setStyleSheet(
            f"background:{color}22; color:{color}; border:1px solid {color}55;"
            "border-radius:3px; padding:3px;"
        )
        layout.addWidget(title)

        self.video = VideoLabel()
        layout.addWidget(self.video, 1)

        stats_row = QHBoxLayout()
        self.fps_lbl = self._stat_label("FPS", "—")
        self.ms_lbl  = self._stat_label("ms",  "—")
        self.obj_lbl = self._stat_label("obj", "—")
        for w in [self.fps_lbl, self.ms_lbl, self.obj_lbl]:
            stats_row.addWidget(w)
        layout.addLayout(stats_row)

    def _stat_label(self, key: str, val: str) -> QLabel:
        lbl = QLabel(f"{key}: {val}")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("Monospace", 8))
        lbl.setStyleSheet(
            "color:#aaa; background:#111; border:1px solid #222;"
            "border-radius:2px; padding:2px 4px;"
        )
        return lbl

    def _start(self):
        self.thread = InferenceThread(
            self.model_info, self.source, self.confidence, self.device
        )
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.error_signal.connect(
            lambda e: self.video.setText(f"Error:\n{e}")
        )
        self.thread.start()

    def _on_frame(self, frame: np.ndarray, stats: dict):
        self.video.update_frame(frame)
        self.fps_lbl.setText(f"FPS: {stats['avg_fps']:.1f}")
        self.ms_lbl.setText(f"ms: {stats['ms']:.1f}")
        self.obj_lbl.setText(f"obj: {stats['objects']}")

    def stop(self):
        if self.thread:
            self.thread.stop()

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)


# ─────────────────────────────────────────────
#  Model Select Dialog
# ─────────────────────────────────────────────

class ModelSelectDialog(QDialog):
    """
    Shows available model sizes grouped by task.
    Each row lists which backends (PT / TRT precisions) are on disk.
    Bottom section has Precision and Device/Runtime radio buttons.
    """

    def __init__(self, available: list[ModelInfo], parent=None,
                 single_select: bool = False):
        super().__init__(parent)
        self.available      = available
        self.single_select  = single_select
        self.selected: list[ModelInfo] = []
        # Returned options
        self.precision  = "fp16"   # fp32 / fp16 / int8
        self.device     = "TensorRT"  # CPU / GPU / TensorRT
        self._build()

    def _build(self):
        self.setMinimumWidth(560)
        self.setMinimumHeight(480)
        self.setStyleSheet("""
            QDialog { background:#0d0d0d; color:#ccc; }
            QLabel  { color:#aaa; }
            QGroupBox {
                color:#777; border:1px solid #2a2a2a;
                border-radius:3px; margin-top:8px; font-size:11px;
            }
            QGroupBox::title { subcontrol-origin:margin; left:8px; }
            QListWidget {
                background:#111; border:1px solid #2a2a2a;
                color:#ccc; font-family:monospace; font-size:12px;
            }
            QListWidget::item { padding:4px 8px; }
            QListWidget::item:selected { background:#2a2a2a; color:#fff; }
            QListWidget::item:hover    { background:#1a1a1a; }
            QRadioButton { color:#aaa; spacing:6px; }
            QRadioButton::indicator {
                width:13px; height:13px;
                border:1px solid #555; border-radius:7px; background:#111;
            }
            QRadioButton::indicator:checked {
                background:#E8622A; border-color:#E8622A;
            }
            QPushButton {
                background:#1e1e1e; color:#aaa;
                border:1px solid #333; border-radius:4px; padding:5px 18px;
            }
            QPushButton:hover { background:#2a2a2a; color:#fff; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Header ──────────────────────────────
        header = QLabel(
            "Choose a task and model size. Options reflect available .pt and "
            "TensorRT engines in the weights folder."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color:#777; font-size:11px;")
        layout.addWidget(header)

        # ── Task selector (shown when all tasks available) ───
        tasks_present = sorted(set(m.task for m in self.available))
        self._task_filter = tasks_present[0] if tasks_present else "Detection"

        if len(tasks_present) > 1:
            task_row = QHBoxLayout()
            task_row.addWidget(QLabel("Task:"))
            self._task_btn_group = QButtonGroup(self)
            task_colors = {"Detection": "#E8622A", "Segmentation": "#2A8CE8", "Pose": "#2AE862"}
            for task in ["Detection", "Segmentation", "Pose"]:
                rb = QRadioButton(task)
                color = task_colors.get(task, "#aaa")
                enabled = task in tasks_present
                rb.setEnabled(enabled)
                rb.setChecked(task == self._task_filter)
                rb.setStyleSheet(
                    f"QRadioButton {{ color: {color if enabled else '#444'}; }}"
                    f"QRadioButton::indicator:checked {{ background:{color}; border-color:{color}; }}"
                )
                rb.toggled.connect(lambda checked, t=task: self._on_task_changed(t) if checked else None)
                self._task_btn_group.addButton(rb)
                task_row.addWidget(rb)
            task_row.addStretch()
            layout.addLayout(task_row)

        # ── Model list ──────────────────────────
        self.list_widget = QListWidget()
        if self.single_select:
            self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        else:
            self.list_widget.setSelectionMode(QListWidget.MultiSelection)

        layout.addWidget(self.list_widget, 1)
        self._populate_list()

        # ── Options ─────────────────────────────
        options_gb = QGroupBox("Options")
        opt_layout = QVBoxLayout(options_gb)
        opt_layout.setSpacing(8)

        # Precision row
        prec_row = QHBoxLayout()
        prec_lbl = QLabel("Precision:")
        prec_lbl.setFixedWidth(110)
        prec_row.addWidget(prec_lbl)

        self._prec_group = QButtonGroup(self)
        for label, val in [("FP32", "fp32"), ("FP16", "fp16"), ("INT8", "int8")]:
            rb = QRadioButton(label)
            if val == "fp16":
                rb.setChecked(True)
            self._prec_group.addButton(rb)
            rb.setProperty("prec_val", val)
            prec_row.addWidget(rb)
        prec_row.addStretch()
        opt_layout.addLayout(prec_row)

        # Device row
        dev_row = QHBoxLayout()
        dev_lbl = QLabel("Device/Runtime:")
        dev_lbl.setFixedWidth(110)
        dev_row.addWidget(dev_lbl)

        self._dev_group = QButtonGroup(self)
        for label, val in [("CPU", "cpu"), ("GPU", "0"), ("TensorRT", "TensorRT")]:
            rb = QRadioButton(label)
            if val == "TensorRT":
                rb.setChecked(True)
            self._dev_group.addButton(rb)
            rb.setProperty("dev_val", val)
            dev_row.addWidget(rb)
        dev_row.addStretch()
        opt_layout.addLayout(dev_row)

        layout.addWidget(options_gb)

        # ── Buttons ─────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setStyleSheet(
            "QPushButton { background:#E8622A; color:white; font-weight:bold;"
            "  border:none; border-radius:4px; padding:6px 24px; }"
            "QPushButton:hover { background:#ff7a3d; }"
        )
        ok_btn.clicked.connect(self._accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _on_task_changed(self, task: str):
        self._task_filter = task
        self._populate_list()

    def _populate_list(self):
        self.list_widget.clear()
        task_models = [m for m in self.available if m.task == self._task_filter]
        task_colors = {"Detection": "#E8622A", "Segmentation": "#2A8CE8", "Pose": "#2AE862"}
        color = task_colors.get(self._task_filter, "#E8622A")

        by_size: dict[str, list] = {}
        for mi in task_models:
            by_size.setdefault(mi.size, []).append(mi)

        suffix = "-seg" if self._task_filter == "Segmentation"                  else "-pose" if self._task_filter == "Pose" else ""

        for size in ["n", "s", "m", "l", "x"]:
            name = f"yolo11{size}{suffix}"
            models_for_size = by_size.get(size, [])

            if models_for_size:
                pt_found  = any(m.backend == "PT"  for m in models_for_size)
                trt_precs = [m.precision for m in models_for_size
                             if m.backend == "TRT" and m.precision]
                tags = []
                if pt_found:
                    tags.append("PT✓")
                if trt_precs:
                    tags.append("TRT: " + ", ".join(trt_precs))
                label = f"{name}    [{',  '.join(tags)}]"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, models_for_size)
                item.setForeground(QColor(color))
            else:
                label = f"{name}    [no files found]"
                item = QListWidgetItem(label)
                item.setForeground(QColor("#444"))
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsEnabled)

            self.list_widget.addItem(item)

        # Pre-select first available row
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).flags() & Qt.ItemIsEnabled:
                self.list_widget.setCurrentRow(i)
                break

    def _accept(self):
        # Read precision
        for btn in self._prec_group.buttons():
            if btn.isChecked():
                self.precision = btn.property("prec_val")
        # Read device
        for btn in self._dev_group.buttons():
            if btn.isChecked():
                self.device = btn.property("dev_val")

        # Collect selected model rows
        sel_items = self.list_widget.selectedItems()
        if not sel_items:
            QMessageBox.warning(self, "No selection", "Please select a model size.")
            return
        if not self.single_select and len(sel_items) > 4:
            QMessageBox.warning(self, "Too many", "Please select at most 4 models.")
            return

        self.selected = []
        for item in sel_items:
            models_for_size: list[ModelInfo] = item.data(Qt.UserRole)
            if not models_for_size:
                continue

            if self.device == "TensorRT":
                # Try to find TRT engine matching chosen precision exactly
                exact = [m for m in models_for_size
                         if m.backend == "TRT" and m.precision == self.precision]
                if exact:
                    chosen = exact[0]
                else:
                    # Fall back to any TRT engine, then PT
                    trt = [m for m in models_for_size if m.backend == "TRT"]
                    if trt:
                        # Warn user their chosen precision isn't available
                        available_precs = [m.precision for m in trt]
                        QMessageBox.warning(
                            self, "Precision not available",
                            f"{self.precision.upper()} engine not found for "
                            f"{item.text().split()[0]}.\n"
                            f"Available TRT precisions: {', '.join(available_precs)}\n"
                            f"Using {trt[0].precision.upper()} instead."
                        )
                        chosen = trt[0]
                    else:
                        chosen = models_for_size[0]
            elif self.device == "cpu":
                pt = [m for m in models_for_size if m.backend == "PT"]
                chosen = pt[0] if pt else models_for_size[0]
            else:  # GPU (PyTorch)
                pt = [m for m in models_for_size if m.backend == "PT"]
                chosen = pt[0] if pt else models_for_size[0]

            self.selected.append(chosen)

        if not self.selected:
            QMessageBox.warning(self, "No files", "No model files found for selection.")
            return
        self.accept()


# ─────────────────────────────────────────────
#  Benchmark Results Table
# ─────────────────────────────────────────────

class BenchTable(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Model", "Avg FPS", "Avg ms", "Min ms", "Max ms",
            "Avg Detections", "Frames"
        ])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 220)
        self.table.setStyleSheet(
            "QTableWidget { background:#0d0d0d; color:#ccc; "
            "  gridline-color:#1e1e1e; border:none; }"
            "QHeaderView::section { background:#1a1a1a; color:#888; "
            "  border:1px solid #2a2a2a; padding:4px; }"
        )
        layout.addWidget(self.table)

    def add_result(self, br: BenchResult):
        row = self.table.rowCount()
        self.table.insertRow(row)
        vals = [
            br.model_label,
            f"{br.avg_fps:.2f}",
            f"{br.avg_ms:.2f}",
            f"{br.min_ms:.2f}",
            f"{br.max_ms:.2f}",
            f"{br.n_objects:.1f}",
            str(br.frames),
        ]
        for col, val in enumerate(vals):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, item)
        # highlight best FPS in green
        self._highlight_best()

    def _highlight_best(self):
        """Color the best (highest) FPS row."""
        rows = self.table.rowCount()
        if rows == 0:
            return
        fps_vals = []
        for r in range(rows):
            try:
                fps_vals.append(float(self.table.item(r, 1).text()))
            except:
                fps_vals.append(0)
        best = max(fps_vals)
        for r in range(rows):
            clr = QColor("#1a3a1a") if fps_vals[r] == best else QColor("#0d0d0d")
            for c in range(self.table.columnCount()):
                it = self.table.item(r, c)
                if it:
                    it.setBackground(clr)

    def clear_results(self):
        self.table.setRowCount(0)


# ─────────────────────────────────────────────
#  Benchmark Tab
# ─────────────────────────────────────────────

class BenchmarkTab(QWidget):
    def __init__(self, available: list[ModelInfo]):
        super().__init__()
        self.available    = available
        self.bench_thread = None
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        # Controls
        ctrl = QGroupBox("Benchmark Settings")
        ctrl.setStyleSheet(
            "QGroupBox { color:#aaa; border:1px solid #2a2a2a; "
            "  border-radius:4px; margin-top:8px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:10px; }"
        )
        ctrl_layout = QGridLayout(ctrl)

        # Source
        ctrl_layout.addWidget(QLabel("Source (0=webcam, or file path):"), 0, 0)
        self.source_input = QComboBox()
        self.source_input.setEditable(True)
        self.source_input.addItems(["/dev/video0", "0", "test.mp4"])
        ctrl_layout.addWidget(self.source_input, 0, 1)

        # Device
        ctrl_layout.addWidget(QLabel("Device:"), 1, 0)
        self.device_box = QComboBox()
        self.device_box.addItems(["0 (GPU/TensorRT)", "cpu"])
        ctrl_layout.addWidget(self.device_box, 1, 1)

        # Confidence
        ctrl_layout.addWidget(QLabel("Confidence:"), 2, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        ctrl_layout.addWidget(self.conf_spin, 2, 1)

        # Frames
        ctrl_layout.addWidget(QLabel("Frames per model:"), 3, 0)
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(10, 1000)
        self.frames_spin.setValue(100)
        ctrl_layout.addWidget(self.frames_spin, 3, 1)

        # Task filter
        ctrl_layout.addWidget(QLabel("Tasks to benchmark:"), 4, 0)
        task_row = QHBoxLayout()
        self.task_checks = {}
        for task in TASK_CONFIGS:
            cb = QCheckBox(task)
            cb.setChecked(True)
            self.task_checks[task] = cb
            task_row.addWidget(cb)
        ctrl_layout.addLayout(task_row, 4, 1)

        layout.addWidget(ctrl)

        # Buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Benchmark")
        self.start_btn.setStyleSheet(
            "QPushButton { background:#E8622A; color:white; font-weight:bold;"
            "  border-radius:4px; padding:8px 20px; }"
            "QPushButton:hover { background:#ff7a3d; }"
        )
        self.start_btn.clicked.connect(self._start_bench)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_bench)
        btn_row.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        # Progress
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet(
            "QProgressBar { background:#111; border:1px solid #333; "
            "  border-radius:3px; text-align:center; color:#aaa; }"
            "QProgressBar::chunk { background:#E8622A; border-radius:3px; }"
        )
        layout.addWidget(self.progress)

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color:#666; font-family:monospace;")
        layout.addWidget(self.status_lbl)

        # Results table
        self.table = BenchTable()
        layout.addWidget(self.table, 1)

    def _start_bench(self):
        src_text = self.source_input.currentText().strip()
        source   = int(src_text) if src_text.isdigit() else src_text
        device   = "0" if "GPU" in self.device_box.currentText() else "cpu"
        conf     = self.conf_spin.value()
        n_frames = self.frames_spin.value()

        selected_tasks = [t for t, cb in self.task_checks.items() if cb.isChecked()]
        models_to_bench = [
            m for m in self.available if m.task in selected_tasks
        ]

        if not models_to_bench:
            QMessageBox.warning(self, "No models", "No models found for selected tasks.")
            return

        self.bench_thread = BenchmarkThread(
            models_to_bench, source, conf, device, n_frames
        )
        self.bench_thread.progress.connect(self._on_progress)
        self.bench_thread.result.connect(self.table.add_result)
        self.bench_thread.finished_all.connect(self._on_done)
        self.bench_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop_bench(self):
        if self.bench_thread:
            self.bench_thread.abort()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.status_lbl.setText(msg)
        if msg.startswith("ERROR"):
            self.status_lbl.setStyleSheet("color:#ff4444; font-family:monospace;")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
        else:
            self.status_lbl.setStyleSheet("color:#666; font-family:monospace;")

    def _on_done(self, results):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if results:
            self.status_lbl.setText(f"Benchmark complete — {len(results)} models tested.")
            self.status_lbl.setStyleSheet("color:#2AE862; font-family:monospace;")
        else:
            self.status_lbl.setStyleSheet("color:#ff4444; font-family:monospace;")

    def _clear(self):
        self.table.clear_results()
        self.progress.setValue(0)
        self.status_lbl.setText("Ready")


# ─────────────────────────────────────────────
#  Live Compare Tab
# ─────────────────────────────────────────────

class CompareTab(QWidget):
    def __init__(self, available: list[ModelInfo]):
        super().__init__()
        self.available  = available
        self.panels: list[ModelPanel] = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        # Toolbar
        bar = QHBoxLayout()
        self.source_box = QComboBox()
        self.source_box.setEditable(True)
        self.source_box.addItems(["/dev/video0", "0", "test.mp4"])
        bar.addWidget(QLabel("Source:"))
        bar.addWidget(self.source_box)

        self.device_box = QComboBox()
        self.device_box.addItems(["0 (GPU/TensorRT)", "cpu"])
        bar.addWidget(QLabel("Device:"))
        bar.addWidget(self.device_box)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(50)
        self.conf_lbl = QLabel("Conf: 0.50")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_lbl.setText(f"Conf: {v/100:.2f}")
        )
        bar.addWidget(QLabel("Confidence:"))
        bar.addWidget(self.conf_slider)
        bar.addWidget(self.conf_lbl)

        self.select_btn = QPushButton("Select Models…")
        self.select_btn.setStyleSheet(
            "QPushButton { background:#2a2a2a; color:#ccc; border:1px solid #444;"
            "  border-radius:4px; padding:6px 12px; }"
            "QPushButton:hover { background:#3a3a3a; }"
        )
        self.select_btn.clicked.connect(self._select_models)
        bar.addWidget(self.select_btn)

        self.stop_btn = QPushButton("Stop All")
        self.stop_btn.setStyleSheet(
            "QPushButton { background:#3a1a1a; color:#E8622A; border:1px solid #E8622A;"
            "  border-radius:4px; padding:6px 12px; }"
        )
        self.stop_btn.clicked.connect(self._stop_all)
        bar.addWidget(self.stop_btn)
        layout.addLayout(bar)

        # Grid for panels
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(6)
        layout.addWidget(self.grid_widget, 1)

        self.hint = QLabel(
            '↑  Click "Select Models…" to start a live side-by-side comparison'
        )
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setStyleSheet("color:#444; font-size:13px; font-family:monospace;")
        self.grid_layout.addWidget(self.hint, 0, 0)

    def _select_models(self):
        dlg = ModelSelectDialog(self.available, self, single_select=False)
        dlg.setWindowTitle("Select Models to Compare")
        if dlg.exec_() == QDialog.Accepted and dlg.selected:
            self._stop_all()
            device = dlg.device if dlg.device in ("cpu", "0") else "0"
            self._launch(dlg.selected, device)

    def _launch(self, models: list[ModelInfo], device: str = "0"):
        # Remove hint
        self.hint.hide()

        src_text = self.source_box.currentText().strip()
        source   = int(src_text) if src_text.isdigit() else src_text
        conf     = self.conf_slider.value() / 100

        cols = 2 if len(models) > 1 else 1
        for i, mi in enumerate(models):
            panel = ModelPanel(mi, source, conf, device)
            self.panels.append(panel)
            self.grid_layout.addWidget(panel, i // cols, i % cols)

    def _stop_all(self):
        for p in self.panels:
            p.stop()
            p.setParent(None)
            p.deleteLater()
        self.panels.clear()
        self.hint.show()


# ─────────────────────────────────────────────
#  Single Inference Tab (matches your screenshot)
# ─────────────────────────────────────────────

class SingleInferenceTab(QWidget):
    def __init__(self, available: list[ModelInfo]):
        super().__init__()
        self.available   = available
        self.thread: Optional[InferenceThread] = None
        self._build()

    def _build(self):
        root = QHBoxLayout(self)

        # ── Left panel ──────────────────────
        left = QWidget()
        left.setFixedWidth(220)
        left.setStyleSheet("background:#111;")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 8, 8)
        lv.setSpacing(8)

        def section(title):
            gb = QGroupBox(title)
            gb.setStyleSheet(
                "QGroupBox { color:#888; border:1px solid #2a2a2a; "
                "  border-radius:3px; margin-top:8px; font-size:11px; }"
                "QGroupBox::title { subcontrol-origin:margin; left:6px; }"
            )
            return gb

        # Task
        task_gb = section("Select Task")
        tv = QVBoxLayout(task_gb)
        self.task_box = QComboBox()
        self.task_box.addItems(list(TASK_CONFIGS.keys()))
        self.task_box.currentTextChanged.connect(self._refresh_model_box)
        tv.addWidget(self.task_box)
        lv.addWidget(task_gb)

        # Model
        model_gb = section("Current Model")
        mv = QVBoxLayout(model_gb)
        self.model_lbl = QLabel("—")
        self.model_lbl.setStyleSheet("color:#E8622A; font-size:10px;")
        mv.addWidget(self.model_lbl)
        self.select_model_btn = QPushButton("Configure Model…")
        self.select_model_btn.clicked.connect(self._configure_model)
        mv.addWidget(self.select_model_btn)
        lv.addWidget(model_gb)

        # Source
        src_gb = section("Select Input Source")
        sv = QVBoxLayout(src_gb)
        self.src_box = QComboBox()
        self.src_box.setEditable(True)
        self.src_box.addItems(["/dev/video0", "0", "test.mp4"])
        sv.addWidget(self.src_box)
        lv.addWidget(src_gb)

        # Device
        dev_gb = section("Device / Runtime")
        dv = QVBoxLayout(dev_gb)
        self.dev_box = QComboBox()
        self.dev_box.addItems(["GPU (TensorRT)", "GPU (PyTorch)", "CPU"])
        dv.addWidget(self.dev_box)
        lv.addWidget(dev_gb)

        # Detection
        det_gb = section("Detection")
        dtv = QVBoxLayout(det_gb)
        det_btns = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet(
            "QPushButton{background:#1a3a1a;color:#2AE862;border:1px solid #2AE862;"
            "border-radius:3px;padding:4px;}"
        )
        self.start_btn.clicked.connect(self._start)
        self.stop_det_btn = QPushButton("Stop")
        self.stop_det_btn.setEnabled(False)
        self.stop_det_btn.clicked.connect(self._stop)
        det_btns.addWidget(self.start_btn)
        det_btns.addWidget(self.stop_det_btn)
        dtv.addLayout(det_btns)
        dtv.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(50)
        dtv.addWidget(self.conf_slider)
        lv.addWidget(det_gb)

        # Stats
        stats_gb = section("Statistics")
        stv = QVBoxLayout(stats_gb)
        self.fps_lbl   = QLabel("FPS: —")
        self.avg_lbl   = QLabel("Avg Inference: —")
        self.objs_lbl  = QLabel("Objects Detected: —")
        for lbl in [self.fps_lbl, self.avg_lbl, self.objs_lbl]:
            lbl.setStyleSheet("color:#aaa; font-size:10px; font-family:monospace;")
            stv.addWidget(lbl)
        lv.addWidget(stats_gb)
        lv.addStretch()

        root.addWidget(left)

        # ── Right: Video ─────────────────────
        self.video = VideoLabel()
        root.addWidget(self.video, 1)

        # Populate model box
        self._refresh_model_box()
        self.current_model: Optional[ModelInfo] = None

    def _refresh_model_box(self):
        task = self.task_box.currentText()
        self._task_models = [m for m in self.available if m.task == task]

    def _configure_model(self):
        task = self.task_box.currentText()
        models = [m for m in self.available if m.task == task]
        if not models:
            QMessageBox.information(self, "No models",
                f"No model files found for {task}.\n"
                "Place .pt or .engine files in the weights/ folder.")
            return

        dlg = ModelSelectDialog(models, self, single_select=True)
        dlg.setWindowTitle(f"Select Model for {task}")
        if dlg.exec_() == QDialog.Accepted and dlg.selected:
            self.current_model = dlg.selected[0]
            self._selected_device    = dlg.device
            self._selected_precision = dlg.precision
            runtime = "TensorRT" if dlg.device == "TensorRT" \
                      else ("GPU" if dlg.device == "0" else "CPU")
            self.model_lbl.setText(
                f"{self.current_model.name}\n"
                f"{runtime} · {dlg.precision.upper()}"
            )

    def _start(self):
        if not self.current_model:
            self._configure_model()
            if not self.current_model:
                return

        src_text = self.src_box.currentText().strip()
        source   = int(src_text) if src_text.isdigit() else src_text
        # /dev/video0 style paths work directly with cv2.VideoCapture
        # Use device from model dialog if available, else fall back to UI dropdown
        device = getattr(self, "_selected_device", None)
        if device is None or device == "TensorRT":
            device = "0"
        elif device not in ("cpu", "0"):
            device = "0"
        conf = self.conf_slider.value() / 100

        self.thread = InferenceThread(self.current_model, source, conf, device)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.error_signal.connect(
            lambda e: self.video.setText(f"Error:\n{e}")
        )
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_det_btn.setEnabled(True)

    def _stop(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.start_btn.setEnabled(True)
        self.stop_det_btn.setEnabled(False)

    def _on_frame(self, frame: np.ndarray, stats: dict):
        self.video.update_frame(frame)
        self.fps_lbl.setText(f"FPS: {stats['avg_fps']:.2f}")
        self.avg_lbl.setText(f"Avg Inference: {stats['ms']:.1f} ms")
        self.objs_lbl.setText(f"Objects Detected: {stats['objects']}")


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO11 Model Comparison — Jetson Orin Nano")
        self.resize(1280, 800)
        self._apply_dark_theme()

        available = discover_models()
        if not available:
            QMessageBox.warning(
                self, "No model files found",
                "No .pt or .engine files found.\n\n"
                "Put your YOLO11 weights in ./weights/ and restart.\n"
                "Example: weights/yolo11n.pt, weights/yolo11s-seg.engine …\n\n"
                "The app will still open but inference won't work."
            )

        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabWidget::pane { border:1px solid #2a2a2a; }"
            "QTabBar::tab { background:#1a1a1a; color:#666; padding:8px 16px; "
            "  border:1px solid #2a2a2a; margin-right:2px; }"
            "QTabBar::tab:selected { background:#2a2a2a; color:#E8622A; "
            "  border-bottom:2px solid #E8622A; }"
        )

        tabs.addTab(SingleInferenceTab(available), "🎯  Single Inference")
        tabs.addTab(CompareTab(available),         "⚡  Live Compare")
        tabs.addTab(BenchmarkTab(available),       "📊  Benchmark")

        self.setCentralWidget(tabs)

        # Status bar
        self.statusBar().setStyleSheet("background:#111; color:#555;")
        n_pt  = sum(1 for m in available if m.backend == "PT")
        n_trt = sum(1 for m in available if m.backend == "TRT")
        self.statusBar().showMessage(
            f"Found {len(available)} models ({n_pt} PT · {n_trt} TRT)  |  "
            "Weights dir: " + str(WEIGHTS_DIR.resolve())
        )

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0d0d0d;
                color: #cccccc;
                font-family: 'DejaVu Sans', 'Segoe UI', sans-serif;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background: #1a1a1a;
                color: #ccc;
                border: 1px solid #333;
                border-radius: 3px;
                padding: 4px;
            }
            QComboBox::drop-down { border: none; }
            QSlider::groove:horizontal {
                background: #222; height: 4px; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #E8622A; width: 12px; height: 12px;
                margin: -4px 0; border-radius: 6px;
            }
            QPushButton {
                background: #1e1e1e;
                color: #aaa;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 5px 12px;
            }
            QPushButton:hover { background: #2a2a2a; color: #fff; }
            QPushButton:disabled { color: #444; border-color: #222; }
            QLabel { color: #aaa; }
            QListWidget {
                background: #111;
                border: 1px solid #2a2a2a;
                color: #ccc;
            }
            QListWidget::item:selected { background: #2a2a2a; }
            QScrollBar:vertical {
                background: #111; width: 8px;
            }
            QScrollBar::handle:vertical { background: #333; border-radius: 4px; }
        """)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO11 Comparison")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
