"""
ManualManager for SmartLabelingApp (PyQt6 Version)
- Manual labeling with PyQt6 interface
- Click and drag to draw bounding boxes
- Multi-class support with visual class selector
- Stores manual boxes with class information
"""

import hashlib
from datetime import datetime
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QRadioButton, QButtonGroup,
                              QScrollArea, QWidget, QFrame,
                              QMessageBox)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QPen, QColor

def default_color_for_name(name: str) -> str:
    """Deterministic color for a class name (simple hash -> hex)."""
    h = int(hashlib.md5(name.encode('utf-8')).hexdigest()[:6], 16)
    return f"#{h:06x}"

def safe_class_name(name: str) -> str:
    """Clean class name by removing extra spaces."""
    return ' '.join(name.strip().split())

# DrawingOverlay class
class DrawingOverlay(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.rect = None
        self.color = QColor("red")

    def paintEvent(self, event):
        if not self.rect:
            return
        painter = QPainter(self)
        pen = QPen(self.color, 3)
        painter.setPen(pen)
        painter.drawRect(self.rect)

class ManualToolbox(QDialog):
    """Floating toolbox for manual labeling with class selection."""
    
    def __init__(self, parent, classes, current_class, on_done, on_exit):
        super().__init__(parent)
        self.setWindowTitle("Manual Labeling")
        self.setMinimumSize(220, 400)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.on_done = on_done
        self.on_exit = on_exit
        self.current_class = current_class
        self.setup_ui(classes)
        
    def setup_ui(self, classes):
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Select Class:")
        header.setStyleSheet("font-size: 12px; font-weight: bold; padding: 5px;")
        layout.addWidget(header)
        
        # Scrollable class list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #ccc; }")
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(2)
        
        self.button_group = QButtonGroup(self)
        
        for cls in classes:
            color = default_color_for_name(cls)
            frame = QFrame()
            frame_layout = QHBoxLayout(frame)
            frame_layout.setContentsMargins(5, 2, 5, 2)
            
            # Color indicator
            color_label = QLabel("  ")
            color_label.setStyleSheet(f"background-color: {color}; border: 1px solid #333;")
            color_label.setFixedSize(20, 20)
            frame_layout.addWidget(color_label)
            
            # Radio button
            radio = QRadioButton(cls)
            if cls == self.current_class:
                radio.setChecked(True)
            radio.toggled.connect(lambda checked, c=cls: self.on_class_changed(c) if checked else None)
            self.button_group.addButton(radio)
            frame_layout.addWidget(radio)
            frame_layout.addStretch()
            
            scroll_layout.addWidget(frame)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Instructions
        info = QLabel("• Click and drag to draw boxes\n• Draw multiple boxes\n• Click Done when finished")
        info.setStyleSheet("color: #555; font-size: 10px; padding: 8px; background-color: #f0f0f0;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Done button
        done_btn = QPushButton("✓ Done")
        done_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; font-size: 12px; font-weight: bold;")
        done_btn.clicked.connect(self.on_done)
        layout.addWidget(done_btn)
        
        # Exit manual mode button
        exit_btn = QPushButton("✕ Exit Manual Mode")
        exit_btn.setStyleSheet(
            "background-color: #c0392b; color: white; padding: 8px; font-size: 11px;"
        )
        exit_btn.clicked.connect(self.on_exit)
        layout.addWidget(exit_btn)
        
        self.setLayout(layout)
    
    def on_class_changed(self, cls):
        self.current_class = cls
    
    def get_selected_class(self):
        return self.current_class

class ManualManager:
    """Manages manual labeling with drawing capabilities."""
    
    def __init__(self, host):
        self.host = host
        
        # Ensure host properties exist
        if not hasattr(host, 'class_samples'):
            host.class_samples = {}
        if not hasattr(host, 'custom_classes'):
            host.custom_classes = []
        if not hasattr(host, 'min_training_samples'):
            host.min_training_samples = 30
        
        # Drawing state
        self._active = False
        self._start_point = None
        self._overlay = None
        self._manual_boxes = []  # [x1, y1, x2, y2, conf, class_name]
        self._current_manual_class = None
        self._toolbox = None
        
        # Bind boxes to image path
        self._active_image_path = None
        
    def start_manual_labeling(self):
        """Begin manual labeling mode."""
        if self._active:
            return  # manual mode already ON

        if not self.host.current_image:
            QMessageBox.warning(self.host, "No Image", "Load an image first.")
            return

        classes = getattr(self.host, 'selected_classes', [])
        if not classes:
            QMessageBox.warning(self.host, "No Class", "Select at least one class first.")
            return

        self._active = True
        self._current_manual_class = classes[0]
        self._active_image_path = str(self.host.current_image_path)
        self._manual_boxes = []

        if not hasattr(self, "_original_mouse_press"):
            self._original_mouse_press = self.host.canvas_label.mousePressEvent
            self._original_mouse_move = self.host.canvas_label.mouseMoveEvent
            self._original_mouse_release = self.host.canvas_label.mouseReleaseEvent

        self._bind_mouse()
        self._create_overlay()
        self._create_toolbox(classes)
        
        # ✅ FIX 1: Update current class from toolbox after creation
        if self._toolbox:
            self._current_manual_class = self._toolbox.get_selected_class()

    def on_image_changed(self):
        """Call this when host changes to next image."""
        if not self._active:
            return

        self._manual_boxes.clear()
        self._start_point = None
        self._active_image_path = str(self.host.current_image_path)
        self._reset_overlay()

    def _bind_mouse(self):
        self.host.canvas_label.mousePressEvent = self._on_mouse_press
        self.host.canvas_label.mouseMoveEvent = self._on_mouse_move
        self.host.canvas_label.mouseReleaseEvent = self._on_mouse_release
        self.host.canvas_label.setMouseTracking(True)
        self.host.canvas_label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.host.canvas_label.setFocus()

    def _unbind_mouse(self):
        if hasattr(self, '_original_mouse_press'):
            self.host.canvas_label.mousePressEvent = self._original_mouse_press
            self.host.canvas_label.mouseMoveEvent = self._original_mouse_move
            self.host.canvas_label.mouseReleaseEvent = self._original_mouse_release
        self.host.canvas_label.setMouseTracking(False)

    def _create_overlay(self):
        if self._overlay:
            self._overlay.deleteLater()
        self._overlay = DrawingOverlay(self.host.canvas_label)
        self._overlay.resize(self.host.canvas_label.size())
        self._overlay.show()

    def _reset_overlay(self):
        if self._overlay:
            self._overlay.rect = None
            self._overlay.update()

    def _create_toolbox(self, classes):
        if self._toolbox:
            return
        self._toolbox = ManualToolbox(
            self.host,
            classes,
            self._current_manual_class,
            self.finish_manual_labeling,
            self._cleanup
        )
        self._toolbox.show()
        self._toolbox.raise_()
        self._toolbox.activateWindow()
    
    def _on_mouse_press(self, event):
        """Handle mouse press to start drawing."""
        if not self._active:
            return
        
        # ✅ UPDATE CLASS FROM TOOLBOX ON PRESS (ensure latest class)
        if self._toolbox:
            self._current_manual_class = self._toolbox.get_selected_class()
        
        self._start_point = event.pos()
        if self._overlay:
            self._overlay.rect = QRect(self._start_point, self._start_point)
            self._overlay.update()
    
    def _on_mouse_move(self, event):
        """Handle mouse drag to draw rectangle."""
        if not self._active or not self._start_point:
            return
        
        if self._overlay:
            self._overlay.rect = QRect(self._start_point, event.pos()).normalized()
            self._overlay.update()
    
    def _on_mouse_release(self, event):
        """Handle mouse release to finish drawing."""
        if not self._active or not self._start_point:
            return
        
        # Get current class from toolbox
        if self._toolbox:
            self._current_manual_class = self._toolbox.get_selected_class()
        
        # Convert display coordinates to original image coordinates
        rect = QRect(self._start_point, event.pos()).normalized()
        
        # Critical check: Ensure scale_factor is available
        if not self.host.current_image or not hasattr(self.host, 'scale_factor') or self.host.scale_factor is None:
            self._start_point = None
            if self._overlay:
                self._overlay.rect = None
                self._overlay.update()
            return
        
        # Get the pixmap from canvas
        pixmap = self.host.canvas_label.pixmap()
        if not pixmap:
            self._start_point = None
            if self._overlay:
                self._overlay.rect = None
                self._overlay.update()
            return
        
        # Calculate offset (canvas is centered)
        canvas_w = self.host.canvas_label.width()
        canvas_h = self.host.canvas_label.height()
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()
        
        offset_x = (canvas_w - pixmap_w) / 2
        offset_y = (canvas_h - pixmap_h) / 2
        
        # Use stored scale_factor
        scale = self.host.scale_factor
        x1 = (rect.x() - offset_x) / scale
        y1 = (rect.y() - offset_y) / scale
        x2 = (rect.right() - offset_x) / scale
        y2 = (rect.bottom() - offset_y) / scale
        
        # Clamp to image bounds
        img_w, img_h = self.host.current_image.size
        x1 = max(0, min(img_w, x1))
        y1 = max(0, min(img_h, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))
        
        # Check if box is valid
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            conf = 100.0
            # ✅ USE SAFE CLASS NAME FOR CONSISTENT COLORS
            cls_name = safe_class_name(self._current_manual_class or "manual")
            self._manual_boxes.append([x1, y1, x2, y2, conf, cls_name])
        
        # Clear overlay
        self._start_point = None
        if self._overlay:
            self._overlay.rect = None
            self._overlay.update()
        
        # Redraw with saved boxes
        self._redraw_with_boxes()
    
    def _redraw_with_boxes(self):
        """Redraw image with all manual boxes."""
        if not self.host.current_image:
            return
        
        # Extra safety: Bind boxes to image path
        current_image_path = str(self.host.current_image_path)
        if self._active_image_path != current_image_path:
            return  # Don't draw boxes from wrong image
        
        # Convert manual boxes to detection format for display
        detections = []
        for box in self._manual_boxes:
            detections.append({
                'bbox': [box[0], box[1], box[2], box[3]],
                'confidence': box[4],
                'class': box[5]
            })
        
        # Redisplay with boxes
        self.host.display_image(detections)
    
    def finish_manual_labeling(self):
        """Save manual boxes and advance to next image while keeping manual mode active."""
        if not self._manual_boxes:
            # Still advance to next image but keep manual mode
            self.host.current_index += 1
            self.host.process_next()
            # Call on_image_changed to update state
            self.on_image_changed()
            return
        
        # Save to host.labels
        img_path = str(self.host.current_image_path)
        entry = self.host.labels.get(img_path, {})
        dets = entry.get('detections', [])
        
        for box in self._manual_boxes:
            # ✅ USE SAFE CLASS NAME FOR CONSISTENT STORAGE
            cls_name = safe_class_name(box[5])
            dets.append({
                'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                'confidence': float(box[4]),
                'class': cls_name,
                'manual': True
            })
        
        entry['detections'] = dets
        
        # Store metadata
        try:
            w, h = self.host.current_image.size
            entry['image_width'] = w
            entry['image_height'] = h
        except Exception:
            pass
        
        entry['timestamp'] = datetime.now().isoformat()
        self.host.labels[img_path] = entry
        
        # Update training counters
        for box in self._manual_boxes:
            # ✅ USE SAFE CLASS NAME FOR CONSISTENT COUNTING
            cls_name = safe_class_name(box[5])
            if cls_name in getattr(self.host, 'custom_classes', []):
                self.host.class_samples[cls_name] = self.host.class_samples.get(cls_name, 0) + 1
        
        # Save and update
        try:
            self.host.save_autosave()
            self.host.update_stats()
        except Exception:
            pass
        
        QMessageBox.information(self.host, "Saved", 
                               f"Saved {len(self._manual_boxes)} manual boxes.")
        
        # Save done, but KEEP manual mode alive
        self._manual_boxes = []
        self._start_point = None
        self._active_image_path = None

        self.host.current_index += 1
        self.host.process_next()
        # Call on_image_changed to update state
        self.on_image_changed()
    
    def _cleanup(self):
        """Complete cleanup when manual mode is explicitly stopped."""
        self._active = False
        
        # Hard reset manual state
        self._manual_boxes = []
        self._start_point = None
        self._active_image_path = None
        
        self._unbind_mouse()
        
        # Cleanup overlay
        if self._overlay:
            self._overlay.deleteLater()
            self._overlay = None
        
        # Exit manual mode completely - close toolbox
        if self._toolbox:
            self._toolbox.close()
            self._toolbox = None
    
    def get_persist_data(self):
        """Return class list and samples for autosave."""
        return {
            'custom_classes': list(getattr(self.host, 'custom_classes', [])),
            'class_samples': dict(getattr(self.host, 'class_samples', {}))
        }
    
    def load_persist_data(self, data):
        """Load custom classes and samples from autosave."""
        if not data:
            return
        
        cc = data.get('custom_classes', [])
        cs = data.get('class_samples', {})
        
        for c in cc:
            # ✅ USE SAFE CLASS NAME FOR CONSISTENT STORAGE
            clean_c = safe_class_name(c)
            if clean_c not in getattr(self.host, 'custom_classes', []):
                self.host.custom_classes.append(clean_c)
        
        for k, v in cs.items():
            # ✅ USE SAFE CLASS NAME FOR CONSISTENT STORAGE
            clean_k = safe_class_name(k)
            self.host.class_samples[clean_k] = v