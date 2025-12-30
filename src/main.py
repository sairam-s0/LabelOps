# main.py -- Smart Labeling v2.3 (Multi-Class Detection) - PyQt6 Version
"""
Smart Labeling v2.3 - Multi-class detection support with PyQt6
- Multiple class selection with dialog UI
- Color-coded bounding boxes per class
- Class names stored per detection
- Modern PyQt6 interface
- FIXED: COCO export, image safety, race conditions, class loading
"""
#import path handling
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from PyQt6.QtCore import pyqtSignal, QObject


import sys, os, shutil, hashlib, tempfile, time, random, json, threading, traceback, atexit
from datetime import datetime
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog,
                              QMessageBox, QDialog, QListWidget, QLineEdit, QFrame,
                              QAbstractItemView)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QKeySequence, QShortcut
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from core.training_orchestrator import TrainingOrchestrator
from core.replay_buffer import ReplayBuffer
from core.data_manager import DataManager
from core.model_manager import ModelManager

# Try to import ManualManager from features; fall back to no-op stub if absent
try:
    from features.manual import ManualManager
except Exception:
    class ManualManager:
        def __init__(self, app):
            self.app = app
        def get_persist_data(self):
            return {}
        def load_persist_data(self, data):
            return
        def start_manual_labeling(self):
            QMessageBox.information(None, "Manual", "Manual labeling module not found")

# ============ WORKER PROCESS ============

_WORKER_MODEL = None

def init_worker(weights, device):
    """Initializer for ProcessPoolExecutor worker process: load model once there."""
    global _WORKER_MODEL
    try:
        from ultralytics import YOLO
        print(f"[Worker] Loading {weights} on {device}...")
        _WORKER_MODEL = YOLO(weights)
        if device == 'cuda':
            try:
                _WORKER_MODEL.to('cuda')
            except Exception:
                pass
        print("[Worker] Ready")
    except Exception as e:
        print(f"[Worker] Init failed: {e}")

def detect_worker(args):
    """Worker detection function with multi-class support."""
    global _WORKER_MODEL
    if not _WORKER_MODEL:
        return {'error': 'Model not loaded in worker', 'detections': []}

    img_path, class_names, thresh, max_dim = args
    temp_fd, temp_path = None, None
    try:
        from PIL import Image
        import numpy as _np

        img = Image.open(img_path)
        orig_size = img.size
        scale = 1.0
        detect_path = img_path

        if max(orig_size) > max_dim:
            scale = max_dim / max(orig_size)
            new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
            img_r = img.resize(new_size, Image.Resampling.LANCZOS)
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            img_r.save(temp_path)
            img_r.close()
            detect_path = temp_path

        img.close()

        results = _WORKER_MODEL(detect_path, verbose=False)

        matching = []
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes:
            boxes = results[0].boxes
            model_names = getattr(_WORKER_MODEL, 'names', {})
            for i in range(len(boxes)):
                try:
                    box = boxes[i]
                    try:
                        cls_t = box.cls
                        if hasattr(cls_t, 'cpu'):
                            cls_id = int(cls_t.cpu().item())
                        else:
                            cls_id = int(cls_t)
                    except Exception:
                        cls_id = int(box.cls)

                    try:
                        conf_t = box.conf
                        if hasattr(conf_t, 'cpu'):
                            conf = float(conf_t.cpu().item()) * 100.0
                        else:
                            conf = float(conf_t) * 100.0
                    except Exception:
                        conf = float(box.conf) * 100.0

                    try:
                        xy = box.xyxy
                        if hasattr(xy, 'cpu'):
                            arr = xy.cpu().numpy()
                        else:
                            arr = _np.array(xy)
                        arr_flat = _np.asarray(arr).reshape(-1)[:4].astype(float)
                        bbox = arr_flat.tolist()
                    except Exception:
                        bbox = [0.0, 0.0, 0.0, 0.0]

                    name = model_names.get(cls_id, None) if isinstance(model_names, dict) else None
                    if name in class_names and conf >= thresh:
                        if scale != 1.0 and scale > 0:
                            bbox = [c / scale for c in bbox]
                        matching.append({
                            'bbox': bbox, 
                            'confidence': conf,
                            'class': name
                        })
                except Exception:
                    continue

        return {'error': None, 'detections': matching}

    except Exception as e:
        return {'error': f'Worker exception: {e}', 'detections': []}
    finally:
        try:
            if temp_fd:
                os.close(temp_fd)
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass

# ============ HELPER FUNCTIONS ============

def default_color_for_name(name: str) -> str:
    """Generate consistent color for class name using hash."""
    h = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
    return f"#{h:06x}"

# ============ CLASS SELECTOR DIALOG ============

class ClassSelectorDialog(QDialog):
    

    def __init__(self, parent, all_classes, selected_classes):
       
        super().__init__(parent)
        self.setWindowTitle("Select Classes for Detection")
        self.setMinimumSize(450, 550)
        self.all_classes = all_classes
        self.selected_classes = selected_classes.copy()

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        header = QLabel("Select one or more classes:")
        header.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(header)

        self.listbox = QListWidget()
        self.listbox.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for cls in self.all_classes:
            self.listbox.addItem(cls)
        for i in range(self.listbox.count()):
            if self.listbox.item(i).text() in self.selected_classes:
                self.listbox.item(i).setSelected(True)
        layout.addWidget(self.listbox)

        add_frame = QHBoxLayout()
        self.new_class_input = QLineEdit()
        self.new_class_input.setPlaceholderText("Add custom class...")
        add_btn = QPushButton("+")
        add_btn.setMaximumWidth(40)
        add_btn.setStyleSheet("background-color: #3498db; color: white;")
        add_btn.clicked.connect(self.add_custom_class)
        add_frame.addWidget(self.new_class_input)
        add_frame.addWidget(add_btn)
        layout.addLayout(add_frame)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 8px 20px;")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: #95a5a6; color: white; padding: 8px 20px;")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def add_custom_class(self):
        val = self.new_class_input.text().strip()
        if val and val not in self.all_classes:
            self.all_classes.append(val)
            self.all_classes.sort()
            self.listbox.clear()
            for cls in self.all_classes:
                self.listbox.addItem(cls)
            for i in range(self.listbox.count()):
                if self.listbox.item(i).text() in self.selected_classes:
                    self.listbox.item(i).setSelected(True)
            self.new_class_input.clear()

    def get_selected(self):
        return [item.text() for item in self.listbox.selectedItems()]

# ============ MAIN APP ============

class SmartLabelingApp(QMainWindow):
    result_ready = pyqtSignal(list)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Labeling v2.3 - Multi-Class (PyQt6)")
        self.setGeometry(100, 100, 1200, 850)

        # Config
        self.weights = 'yolov8m.pt'
        
        # ✅ LOAD CLASS NAMES IN MAIN THREAD (STEP 2)
        # Load class names from model metadata (fast, no GPU needed)
        try:
            from ultralytics import YOLO
            print("Loading model class names from metadata...")
            # This loads only model config, not full weights onto GPU
            temp_model = YOLO(self.weights)
            self.coco_classes = list(temp_model.names.values())
            print(f"✅ Loaded {len(self.coco_classes)} classes.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load class names from {self.weights}: {e}")
            # Fallback to common COCO classes if model fails
            self.coco_classes = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ]

        self.data_manager = DataManager("labels.json")
        self.replay_buffer = ReplayBuffer(max_size=200)
        self.model_manager = ModelManager("models")


        self.has_gpu = self.detect_gpu()
        self.device = 'cuda' if self.has_gpu else 'cpu'
        self.max_dim = 1600
        self.threshold = 70
        self.qa_rate = 0.01

        self.orchestrator = TrainingOrchestrator(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
            replay_buffer=self.replay_buffer,
            min_samples=30,
            num_gpus=1 if self.has_gpu else 0
        )

        # State
        self.executor = None
        self.worker_ready = False
        self.image_files = []
        self.current_index = 0
        self.labels = {}
        self.auto_accepted_log = []
        self.detection_errors = []
        self.current_image = None
        self.current_image_path = None
        self._current_img_size = (0, 0)  # ✅ Store size at detection time
        self.scale_factor = None
        self.current_detections = []

        # Autosave
        self.autosave_file = None
        self.autosave_lock = threading.Lock()

        # Multi-class
        self.selected_classes = []
        self.custom_classes = []
        self.show_manual_instructions = True
        self.class_samples = {}
        self.min_training_samples = 30

        self.setup_ui()
        self.result_ready.connect(self.handle_result)

         
        self.orchestrator.on_status_change = self._on_training_status_change
        self.orchestrator.on_training_complete = self._on_training_complete
        self.orchestrator.on_training_failed = self._on_training_failed

         
        self.training_timer = QTimer()
        self.training_timer.timeout.connect(self._check_training_status)
        self.training_timer.start(2000)  # Check every 2 seconds

        self.manual = ManualManager(self)
        self.init_worker()
        atexit.register(self.cleanup)

    @property
    def selected_class(self):
        return self.selected_classes[0] if self.selected_classes else None

    def detect_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def update_training_progress(self):
        pass

    # ✅ SIMPLIFIED init_worker (STEP 3)
    def init_worker(self):
        """Start worker process (only loads model — classes already known)."""
        def init():
            try:
                self.executor = ProcessPoolExecutor(
                    max_workers=1,
                    initializer=init_worker,
                    initargs=(self.weights, self.device)
                )
                # ✅ Classes already loaded in main thread
                self.worker_ready = True
                QTimer.singleShot(0, self.on_worker_ready)
            except Exception as e:
                print(f"[init_worker] Error: {e}")
                # ✅ Fixed lambda capture
                QTimer.singleShot(0, lambda err=e: QMessageBox.critical(self, "Error", f"Worker init failed: {err}"))
        threading.Thread(target=init, daemon=True).start()

    def on_worker_ready(self):
        QMessageBox.information(self, "Ready", "Worker process ready and model loaded.")
        self.class_button.setEnabled(True)

    def cleanup(self):
        print("[Cleanup] shutting down executor...")
        try:
            if self.executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            if self.current_image:
                self.current_image.close()
        except Exception:
            pass
        print("[Cleanup] done")

    def closeEvent(self, event):
        try:
            self.save_autosave()
        except Exception:
            pass
        self.cleanup()
        event.accept()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top controls
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("background-color: #2c3e50; padding: 10px;")
        ctrl_layout = QHBoxLayout(ctrl_frame)

        folder_btn = QPushButton("📁 Folder")
        folder_btn.setStyleSheet("background-color: #3498db; color: white; padding: 10px 20px; font-size: 12px;")
        folder_btn.clicked.connect(self.select_folder)
        ctrl_layout.addWidget(folder_btn)

        class_label = QLabel("Classes:")
        class_label.setStyleSheet("color: white; font-size: 12px;")
        ctrl_layout.addWidget(class_label)

        self.class_button = QPushButton("Select Classes")
        self.class_button.setStyleSheet("background-color: #9b59b6; color: white; padding: 8px; font-size: 10px;")
        self.class_button.setEnabled(False)  # Initially disabled until worker ready
        self.class_button.clicked.connect(self.open_class_selector)
        ctrl_layout.addWidget(self.class_button)

        self.class_label = QLabel("None selected")
        self.class_label.setStyleSheet("color: #95a5a6; font-size: 9px;")
        ctrl_layout.addWidget(self.class_label)

        thresh_label = QLabel("Threshold:")
        thresh_label.setStyleSheet("color: white; font-size: 12px;")
        ctrl_layout.addWidget(thresh_label)

        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setMinimum(0)
        self.thresh_slider.setMaximum(100)
        self.thresh_slider.setValue(70)
        self.thresh_slider.setFixedWidth(160)
        ctrl_layout.addWidget(self.thresh_slider)

        self.start_btn = QPushButton("▶️ Start")
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px 20px; font-weight: bold; font-size: 12px;")
        self.start_btn.clicked.connect(self.start_labeling)
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addStretch()

        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("color: white; font-size: 10px;")
        ctrl_layout.addWidget(self.progress_label)

        gpu_label = QLabel("🟢 GPU" if self.has_gpu else "⚪ CPU")
        gpu_label.setStyleSheet(f"color: {'#27ae60' if self.has_gpu else '#95a5a6'}; font-size: 9px;")
        ctrl_layout.addWidget(gpu_label)

        #Training status display
        self.training_status_label = QLabel("Training: Not started")
        self.training_status_label.setStyleSheet("color: #95a5a6; font-size: 10px;")
        ctrl_layout.addWidget(self.training_status_label)


        main_layout.addWidget(ctrl_frame)

        # Canvas
        self.canvas_label = QLabel()
        self.canvas_label.setStyleSheet("background-color: #34495e; color: white;")
        self.canvas_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_label.setMinimumSize(800, 600)
        main_layout.addWidget(self.canvas_label, stretch=1)

        # Action buttons
        btn_frame = QFrame()
        btn_frame.setStyleSheet("background-color: #ecf0f1; padding: 10px;")
        btn_layout = QHBoxLayout(btn_frame)

        for text, color, cb in [
            ("✓ Accept (A)", "#27ae60", self.accept),
            ("✗ Reject (R)", "#e74c3c", self.reject),
            ("⏭️ Skip (N)", "#95a5a6", self.skip),
            ("✏️ Manual (M)", "#f39c12", lambda: self.manual.start_manual_labeling()),
            ("📋 Auto Log", "#9b59b6", self.show_log)
        ]:
            btn = QPushButton(text)
            btn.setStyleSheet(f"background-color: {color}; color: white; padding: 8px; font-size: 11px;")
            btn.clicked.connect(cb)
            btn_layout.addWidget(btn)

        main_layout.addWidget(btn_frame)

        # Export
        export_frame = QFrame()
        export_frame.setStyleSheet("background-color: #ecf0f1; padding: 10px;")
        export_layout = QHBoxLayout(export_frame)
        export_layout.addStretch()
        for text, color, fmt in [("💾 JSON", "#9b59b6", 'json'), ("💾 COCO", "#8e44ad", 'coco')]:
            btn = QPushButton(text)
            btn.setStyleSheet(f"background-color: {color}; color: white; padding: 8px 15px; font-size: 11px;")
            btn.clicked.connect(lambda _, f=fmt: self.export(f))
            export_layout.addWidget(btn)
        export_layout.addStretch()
        promote_btn = QPushButton("Promote Shadow")
        promote_btn.setStyleSheet("background-color: #e67e22; color: white; padding: 8px 15px; font-size: 11px;")
        promote_btn.clicked.connect(self.promote_shadow_model)
        export_layout.addWidget(promote_btn)

        main_layout.addWidget(export_frame)


        # Shortcuts
        for key, cb in [('a', self.accept), ('r', self.reject), ('n', self.skip), ('m', lambda: self.manual.start_manual_labeling())]:
            QShortcut(QKeySequence(key), self).activated.connect(cb)

    def open_class_selector(self):
        # ✅ Now self.coco_classes is always populated, so this check passes
        if not self.coco_classes:
            QMessageBox.warning(self, "Not Ready", "Wait for model to load")
            return
        all_classes = sorted(self.coco_classes + self.custom_classes) or ["person", "car", "bicycle"]
        dialog = ClassSelectorDialog(self, all_classes, self.selected_classes)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.selected_classes = dialog.get_selected()
            self.custom_classes = [c for c in dialog.all_classes if c not in self.coco_classes]
            if self.selected_classes:
                display = ", ".join(self.selected_classes[:3])
                if len(self.selected_classes) > 3:
                    display += f" +{len(self.selected_classes)-3} more"
                self.class_label.setText(display)
                self.class_label.setStyleSheet("color: white; font-size: 9px;")
            else:
                self.class_label.setText("None selected")
                self.class_label.setStyleSheet("color: #95a5a6; font-size: 9px;")

    def _ensure_autosave_setup(self, folder_path: Path):
        try:
            self.autosave_file = str(folder_path / "labels_autosave.json")
        except Exception:
            self.autosave_file = None

    def save_autosave(self):
        if not getattr(self, 'autosave_file', None):
            return
        try:
            persist = {
                'labels': self.labels,
                'current_index': self.current_index,
                'auto_accepted_log': self.auto_accepted_log,
                'detection_errors': self.detection_errors,
                'selected_classes': self.selected_classes,
                'saved_at': datetime.now().isoformat()
            }
            if hasattr(self, 'manual') and callable(getattr(self.manual, 'get_persist_data', None)):
                try:
                    persist.update(self.manual.get_persist_data())
                except Exception:
                    pass
            autosave_path = Path(self.autosave_file)
            autosave_path.parent.mkdir(parents=True, exist_ok=True)
            with self.autosave_lock:
                fd, tmp = tempfile.mkstemp(prefix="autosave_", suffix=".tmp", dir=str(autosave_path.parent))
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        json.dump(persist, f, indent=2, ensure_ascii=False)
                    if os.name == 'nt' and autosave_path.exists():
                        try: os.remove(autosave_path)
                        except Exception: pass
                    os.replace(tmp, autosave_path)
                except Exception:
                    try: os.unlink(tmp)
                    except Exception: pass
                    raise
        except Exception as e:
            print(f"[save_autosave] failed: {e}")

    def load_autosave(self):
        if not getattr(self, 'autosave_file', None):
            return
        path = Path(self.autosave_file)
        if not path.exists():
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.labels = data.get('labels', {}) or {}
            self.current_index = int(data.get('current_index', 0) or 0)
            self.auto_accepted_log = data.get('auto_accepted_log', self.auto_accepted_log)
            self.detection_errors = data.get('detection_errors', self.detection_errors)
            saved_classes = data.get('selected_classes', [])
            if saved_classes:
                self.selected_classes = saved_classes
                display = ", ".join(saved_classes[:3])
                if len(saved_classes) > 3:
                    display += f" +{len(saved_classes)-3} more"
                self.class_label.setText(display)
                self.class_label.setStyleSheet("color: white; font-size: 9px;")
            if hasattr(self, 'manual') and callable(getattr(self.manual, 'load_persist_data', None)):
                try:
                    self.manual.load_persist_data(data)
                except Exception:
                    pass
            if self.labels:
                msg = f"Autosave loaded — {len(self.labels)} labeled images. Resuming at index {self.current_index + 1}."
                print("[load_autosave] " + msg)
                self.progress_label.setText("Autosave loaded")
                QMessageBox.information(self, "Autosave", msg)
                self.update_stats()
        except Exception as e:
            print(f"[load_autosave] failed: {e}")

    def update_stats(self):
        total = len(self.image_files) if self.image_files else 0
        labeled = len(self.labels) if self.labels else 0
        self.progress_label.setText(f"Labeled: {labeled}/{total}")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        folder_p = Path(folder)
        self.image_files = list(folder_p.glob("*.jpg")) + list(folder_p.glob("*.png")) + list(folder_p.glob("*.jpeg"))
        self.current_index = 0
        QMessageBox.information(self, "Success", f"Loaded {len(self.image_files)} images")
        self._ensure_autosave_setup(folder_p)
        self.load_autosave()
        self.update_stats()

    def start_labeling(self):
        if not self.worker_ready:
            QMessageBox.critical(self, "Error", "Worker not ready — wait for model to finish loading.")
            return
        if not self.image_files:
            QMessageBox.critical(self, "Error", "Select folder first")
            return
        if not self.selected_classes:
            QMessageBox.critical(self, "Error", "Please select at least one class.")
            return
        self.threshold = self.thresh_slider.value()
        self.current_index = max(0, int(self.current_index))
        self.progress_label.setText("Starting...")
        QTimer.singleShot(150, self.process_next)

    # ✅ FIXED: Don't close current_image here!
    def process_next(self):
        if self.current_image:
            try:
                self.current_image.close()
            except Exception:
                pass
            self.current_image = None
        
        
        if self.current_index >= len(self.image_files):
            QMessageBox.information(self, "Done", "All images processed.")
            self.progress_label.setText("Complete")
            try: self.save_autosave()
            except Exception: pass
            return

        img_path = self.image_files[self.current_index]
        self.current_image_path = img_path
        self.progress_label.setText(f"{self.current_index+1}/{len(self.image_files)}")
        try:
            self.current_image = Image.open(img_path)
            if self.current_image.mode != 'RGB':
                self.current_image = self.current_image.convert('RGB')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open image: {e}")
            self.current_index += 1
            self.process_next()
            return

        threading.Thread(target=lambda: self.run_detect(img_path), daemon=True).start()

    def run_detect(self, img_path):
        try:
            args = (str(img_path), self.selected_classes, self.threshold, self.max_dim)
            future = self.executor.submit(detect_worker, args)
            result = future.result(timeout=30)
            if result.get('error'):
                self.detection_errors.append((img_path, result.get('error')))
            detections = result.get('detections', [])
            self.result_ready.emit(detections)

        except Exception as e:
            print(f"[run_detect] {e}")
            # ✅ This is safe - no variable capture needed
            self.result_ready.emit([])

    # ✅ FIXED: Capture image size at detection time
    def handle_result(self, detections):
        self.current_detections = detections or []
        
        # ✅ Store image size NOW for later saving
        if self.current_image:
            self._current_img_size = self.current_image.size
        else:
            self._current_img_size = (0, 0)
            
        self.display_image(detections)
        if detections:
            max_conf = max((d['confidence'] for d in detections), default=0.0)
            if max_conf >= self.threshold:
                if random.random() >= self.qa_rate:
                    self.save_label(detections, auto=True)
                    self.auto_accepted_log.append(str(self.image_files[self.current_index]))
                    self.current_index += 1
                    try:
                        self.save_autosave()
                        self.update_stats()
                    except Exception:
                        pass
                    QTimer.singleShot(50, self.process_next)
                    return
        self.progress_label.setText("Review required")

    # ✅ FIXED: Add image validity check
    def display_image(self, detections):
        print("display_image called, current_image =", self.current_image)
        if not self.current_image:
            self.canvas_label.setText("No image loaded")
            return

        # ✅ Check if image is still valid
        try:
            _ = self.current_image.size  # Will raise exception if closed/corrupted
        except Exception:
            self.canvas_label.setText("Image closed or corrupted")
            return

        canvas_w = self.canvas_label.width()
        canvas_h = self.canvas_label.height()

        if canvas_w < 100 or canvas_h < 100:
            QTimer.singleShot(50, lambda: self.display_image(detections))
            return

        try:
            img_draw = self.current_image.copy()
            draw = ImageDraw.Draw(img_draw)
            
            for d in (detections or []):
                b = d.get('bbox', [0, 0, 0, 0])
                cls_name = d.get('class', 'unknown')
                conf = d.get('confidence', 0)
                color = default_color_for_name(cls_name)
                iw, ih = img_draw.size
                b = [max(0, min(iw, x)) for x in b]
                draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline=color, width=3)
                draw.text((b[0], max(0, b[1] - 14)), f"{cls_name}: {conf:.1f}%", fill=color)

            iw, ih = img_draw.size
            scale = min(canvas_w / iw, canvas_h / ih, 1.0)
            if scale <= 0:
                scale = 1.0
            self.scale_factor = scale

            new_w = max(1, int(iw * scale))
            new_h = max(1, int(ih * scale))
            img_resized = img_draw.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_rgb = img_resized.convert('RGB')
            raw = img_rgb.tobytes()  # store in local var so GC won't free it immediately
            bytes_per_line = new_w * 3
            qimage = QImage(raw, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
            if qimage.isNull():
                print(f"[display_image] QImage is null (w={new_w}, h={new_h}, len(raw)={len(raw)})")
                self.canvas_label.setText("Failed to create image")
                return

            pixmap = QPixmap.fromImage(qimage.copy())  # copy() to be extra-safe about ownership
            self.canvas_label.setPixmap(pixmap)
            del raw

            img_resized.close()
            img_draw.close()

        except Exception as e:
            print(f"Display error: {e}")
            self.canvas_label.setText(f"Error: {str(e)}")

    # ✅ FIXED: Add image dimensions to labels for COCO export
    def save_label(self, detections, auto=False):
        img_path = str(self.image_files[self.current_index])
        
        # Get image dimensions (use captured size if available)
        img_w, img_h = getattr(self, '_current_img_size', (0, 0))
        
        clean_dets = []
        for d in (detections or []):
            clean_dets.append({
                'bbox': [int(round(x)) for x in d['bbox']],
                'confidence': round(d.get('confidence', 0.0), 2),
                'class': d.get('class', self.selected_classes[0] if self.selected_classes else 'unknown')
            })
        
        self.labels[img_path] = {
            'detections': clean_dets,
            'auto': bool(auto),
            'timestamp': datetime.now().isoformat(),
            'image_width': img_w,    # ✅ For COCO export
            'image_height': img_h    # ✅ For COCO export
        }
        try:
            self.update_stats()
            self.save_autosave()
        except Exception:
            pass

    def accept(self):
        if getattr(self, 'current_detections', None):
            self.save_label(self.current_detections, auto=False)
        self.current_index += 1
        """Modified to include training trigger check."""
        if getattr(self, 'current_detections', None):
            # âœ… MODIFIED: Save with full metadata using data_manager
            if hasattr(self, 'data_manager'):
                img_path = str(self.current_image_path)
                img_w, img_h = getattr(self, '_current_img_size', (0, 0))
                
                # Save using data_manager (includes entropy, dimensions, etc.)
                self.data_manager.save_labels(
                    image_path=img_path,
                    detections=self.current_detections,
                    entropy=getattr(self, 'last_image_entropy', 0.0),
                    img_width=img_w,
                    img_height=img_h
                )
                
                # âœ… ADD: Check if we should trigger training
                self.orchestrator.check_training_trigger()
            else:
                # Fallback to original method if data_manager not available
                self.save_label(self.current_detections, auto=False)
        
        self.current_index += 1
        self.process_next()

    def reject(self):
        try: self.save_autosave()
        except Exception: pass
        self.current_index += 1
        self.process_next()

    def skip(self):
        try: self.save_autosave()
        except Exception: pass
        self.current_index += 1
        self.process_next()

    def show_log(self):
        if not self.auto_accepted_log:
            QMessageBox.information(self, "Log", "No auto-accepted images yet")
            return
        msg = "\n".join([Path(p).name for p in self.auto_accepted_log[-20:]])
        QMessageBox.information(self, "Auto-accepted (last 20)", msg)

    def export(self, fmt):
        if not self.labels:
            QMessageBox.warning(self, "No Data", "No labels to export")
            return
        if fmt == 'json':
            file, _ = QFileDialog.getSaveFileName(self, "Export JSON", "", "JSON Files (*.json)")
            if file:
                with open(file, 'w') as f:
                    json.dump(self.labels, f, indent=2)
                QMessageBox.information(self, "Success", f"Exported {len(self.labels)} labels")
        elif fmt == 'coco':
            file, _ = QFileDialog.getSaveFileName(self, "Export COCO", "", "JSON Files (*.json)")
            if file:
                coco = {'images': [], 'annotations': [], 'categories': []}
                class_to_id = {}
                cat_id = 1
                for img_path, ld in self.labels.items():
                    for det in ld.get('detections', []):
                        cls_name = det.get('class', 'unknown')
                        if cls_name not in class_to_id:
                            class_to_id[cls_name] = cat_id
                            coco['categories'].append({'id': cat_id, 'name': cls_name, 'supercategory': 'object'})
                            cat_id += 1
                img_id = 1
                ann_id = 1
                for img_path, ld in self.labels.items():
                    coco['images'].append({
                        'id': img_id,
                        'file_name': Path(img_path).name,
                        'width': ld.get('image_width', 0),    # ✅ Now available!
                        'height': ld.get('image_height', 0)   # ✅ Now available!
                    })
                    for det in ld.get('detections', []):
                        bbox = det['bbox']
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                        cls_name = det.get('class', 'unknown')
                        cat_id = class_to_id.get(cls_name, 1)
                        coco['annotations'].append({
                            'id': ann_id,
                            'image_id': img_id,
                            'category_id': cat_id,
                            'bbox': [bbox[0], bbox[1], w, h],
                            'area': w * h,
                            'iscrowd': 0
                        })
                        ann_id += 1
                    img_id += 1
                with open(file, 'w') as f:
                    json.dump(coco, f, indent=2)
                QMessageBox.information(self, "Success", f"COCO exported with {len(coco['categories'])} classes")
    
    def _check_training_status(self):
        """Called by timer to update training status."""
        if not hasattr(self, 'orchestrator'):
            return
        
        # Check if training completed
        result = self.orchestrator.check_training_completion()
        # Result handling done via callbacks
        
        # Update status display
        status = self.orchestrator.get_training_status()
        
        if status.get('training'):
            epoch = status.get('epoch', 0)
            total = status.get('total_epochs', 1)
            loss = status.get('loss', 0)
            percent = status.get('percent', 0)
            
            self.training_status_label.setText(
                f"Training: {epoch}/{total} ({percent:.0f}%) Loss: {loss:.4f}"
            )
            self.training_status_label.setStyleSheet("color: #f39c12; font-size: 10px;")
        else:
            # Show queue status
            queue = self.orchestrator.get_queue_status()
            self.training_status_label.setText(
                f"Queue: {queue['queue_size']}/{queue['min_samples']} samples"
            )
            color = "#27ae60" if queue['ready_to_train'] else "#95a5a6"
            self.training_status_label.setStyleSheet(f"color: {color}; font-size: 10px;")
    
    #Callback handlers
    def _on_training_status_change(self, status):
        """Called when training status changes."""
        if status['status'] == 'training_started':
            msg = f"Training started: {status['sample_count']} samples"
            self.progress_label.setText(msg)
            print(f"[Main] {msg}")

        
    def _on_training_complete(self, result):
        """Called when training completes successfully."""
        msg = f"Training complete! Model saved to: {result['save_path']}"
        self.progress_label.setText("Training complete")
        
        QMessageBox.information(
            self,
            "Training Complete",
            f"Shadow model training completed!\n\n"
            f"Samples: {result['sample_count']}\n"
            f"Model: {result['save_path']}\n\n"
            f"You can now promote the shadow model."
        )

        
    def _on_training_failed(self, result):
        """Called when training fails."""
        error = result.get('error', 'Unknown error')
        QMessageBox.critical(
            self,
            "Training Failed",
            f"Shadow model training failed:\n\n{error}\n\n"
            f"Check console for details."
        )
           
    # Promote handler
    def promote_shadow_model(self):
        """Handle shadow model promotion."""
        if not hasattr(self, 'orchestrator'):
            QMessageBox.warning(self, "Error", "Training system not initialized")
            return
        
        # Confirm promotion
        reply = QMessageBox.question(
            self,
            "Promote Shadow Model",
            "Promote shadow model to active?\n\n"
            "This will make it the default for inference.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Attempt promotion
        result = self.orchestrator.promote_shadow_model(validate=True)
        
        if result['success']:
            QMessageBox.information(
                self,
                "Success",
                f"Shadow model promoted!\n\n"
                f"Version: {result['version']}\n"
                f"Path: {result['path']}\n\n"
                f"Restart the app to use the new model."
            )
        elif result.get('requires_confirmation'):
            # Validation failed - ask user
            reply = QMessageBox.question(
                self,
                "Validation Warning",
                f"Model validation detected issues:\n\n{result['error']}\n\n"
                f"Promote anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                result = self.orchestrator.promote_shadow_model(validate=False)
                if result['success']:
                    QMessageBox.information(self, "Success", "Model promoted!")
        else:
            QMessageBox.critical(self, "Error", f"Promotion failed:\n\n{result['error']}")
    
        
    def closeEvent(self, event):
        """Modified to include orchestrator cleanup."""
        try:
            self.save_autosave()
        except Exception:
            pass
        
        # âœ… ADD: Shutdown orchestrator
        if hasattr(self, 'orchestrator'):
            self.orchestrator.shutdown()
        
        self.cleanup()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = SmartLabelingApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()