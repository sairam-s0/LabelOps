# src/core/entropy.py
import math
import numpy as np
import torch

class EntropyCalculator:
    @staticmethod
    def normalized_entropy(probs: list[float]) -> float:
        """
        probs: list of class probabilities (softmax output)
        returns: 0.0 (certain) → 1.0 (max uncertainty)
        """
        if not probs:
            return 0.0

        probs = np.clip(np.array(probs, dtype=float), 1e-9, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = math.log(len(probs))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    @staticmethod
    def from_yolo_output(box_output, num_classes: int = 80) -> float:
        """
        Calculate entropy from raw YOLO box output.
        Handles both logits (needs softmax) and pre-computed probabilities.
        
        Args:
            box_output: YOLO detection result (e.g., from model.predict())
            num_classes: Total number of classes in model (default: 80 for COCO)
        """
        # Option 1: If model outputs direct probabilities (Classify head)
        if hasattr(box_output, 'probs') and box_output.probs is not None:
            probs = box_output.probs.cpu().numpy()
            return EntropyCalculator.normalized_entropy(probs.tolist())
        
        # Option 2: If model outputs logits (Detect head with extended output)
        if hasattr(box_output, 'data') and isinstance(box_output.data, torch.Tensor):
            data = box_output.data
            if data.ndim == 2 and data.shape[1] > 6:
                # Assume format: [x1, y1, x2, y2, conf, cls, ...logits]
                if data.shape[1] == 6 + num_classes:
                    logits = data[:, 6:]  # Extract logits for highest-conf box (or per-box)
                    # Use the first detection's logits for simplicity
                    if logits.shape[0] > 0:
                        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
                        return EntropyCalculator.normalized_entropy(probs.tolist())

        # ❗ Fallback for standard YOLOv8 (only [x,y,x,y,conf,cls] per box)
        # We simulate a probability distribution:
        #   p(winner) = confidence
        #   p(other)  = (1 - confidence) / (num_classes - 1)
        try:
            # YOLOv8 Results object: box_output.boxes contains detection data
            if hasattr(box_output, 'boxes') and box_output.boxes is not None:
                boxes = box_output.boxes  # Ultralytics YOLOv8 Boxes object
                if hasattr(boxes, 'conf') and len(boxes.conf) > 0:
                    # Use the detection with the highest confidence (most representative)
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()
                    max_conf_idx = int(np.argmax(confs))
                    confidence = float(confs[max_conf_idx])
                    pred_class = int(classes[max_conf_idx])

                    # Build synthetic probability distribution
                    if num_classes <= 1:
                        return 0.0

                    probs = np.full(num_classes, (1.0 - confidence) / (num_classes - 1))
                    probs[pred_class] = confidence

                    # Clip to valid range and renormalize (tiny numerical errors possible)
                    probs = np.clip(probs, 1e-9, 1.0)
                    probs /= probs.sum()

                    return EntropyCalculator.normalized_entropy(probs.tolist())
        except (AttributeError, IndexError, ValueError, TypeError) as e:
            # If any step fails, we truly have no data → return 0.0
            pass

        # Absolute fallback: no detection info available
        return 0.0

    @staticmethod
    def image_entropy(detections: list[dict]) -> float:
        """
        Default aggregation: Max entropy across detections
        """
        if not detections:
            return 0.0
        entropies = [d.get("entropy", 0.0) for d in detections]
        return max(entropies, default=0.0)

    @staticmethod
    def aggregate_entropy(detections: list[dict], method='max') -> float:
        """
        Calculate single entropy score for an entire image using different strategies.
        """
        if not detections:
            return 0.0
        
        entropies = [d.get("entropy", 0.0) for d in detections]
        confidences = [d.get('confidence', 0.0) for d in detections]
        
        if method == 'max':
            return max(entropies)
        elif method == 'mean':
            return sum(entropies) / len(entropies)
        elif method == 'weighted':
            total_conf = sum(confidences)
            if total_conf == 0:
                return 0.0
            weighted_sum = sum(e * c for e, c in zip(entropies, confidences))
            return weighted_sum / total_conf
            
        return max(entropies)