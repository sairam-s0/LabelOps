# src/core/inference.py
from ultralytics import YOLO
from .entropy import EntropyCalculator
import torch
import numpy as np

class InferenceEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.num_classes = len(self.model.names)

    def reload(self):
        self.model = YOLO(self.model_path)

    def predict(self, image_path: str, selected_classes: list = None, threshold: float = 0.5):
        """
        Runs inference and calculates entropy using actual model outputs.
        """
        results = self.model(image_path, verbose=False)
        detections = []

        if not results or not results[0].boxes:
            return detections, 0.0

        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            class_name = self.model.names[cls_id]

            # --- CRITICAL FIX: PROBABILITY CALCULATION ---
            if hasattr(box, 'probs') and box.probs is not None:
                # Classification model output
                probs = box.probs.cpu().numpy()
            elif hasattr(box, 'data') and box.data.shape[-1] > 6:
                # Custom Detection model exporting logits
                logits = box.data[5:] # Adjust index based on specific model export
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                # Fallback: Standard YOLOv8 detection only gives us Top-1 Confidence.
                # We construct a synthetic distribution. 
                # Ideally, you want to use a model exported with `agnostic_nms=True` or modified head.
                probs = np.zeros(self.num_classes)
                probs[cls_id] = conf
                remainder = (1.0 - conf) / (self.num_classes - 1)
                probs = np.where(probs == 0, remainder, probs)

            entropy = EntropyCalculator.normalized_entropy(probs.tolist())

            # Filters
            if selected_classes and class_name not in selected_classes:
                continue
            if conf < threshold:
                continue

            detections.append({
                "bbox": box.xyxy[0].tolist(),
                "confidence": conf * 100,
                "class": class_name,
                "entropy": entropy
            })

        return detections, EntropyCalculator.image_entropy(detections)

    def predict_batch(self, image_paths: list) -> list:
        """
        Process multiple images efficiently.
        """
        results = []
        for path in image_paths:
            det, ent = self.predict(path)
            results.append((det, ent))
        return results