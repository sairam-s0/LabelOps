# src/core/shadow_trainer.py
import ray
from ultralytics import YOLO
from pathlib import Path
import tempfile
import shutil
import yaml
from datetime import datetime
import torch

@ray.remote(num_gpus=0.4)
class ShadowTrainer:
    """
    Ray actor for asynchronous model training with replay buffer integration.
    """
    def __init__(self, base_model_path: str, class_mapping: dict, min_samples=30):
        """
        Initialize shadow trainer.
        
        Args:
            base_model_path: Path to base model to fine-tune
            class_mapping: Dict mapping class names to IDs, e.g., {'person': 0, 'car': 1}
            min_samples: Minimum samples required before training
        """
        print("[Shadow] Initializing Worker...")
        self.base_model_path = base_model_path
        self.class_mapping = class_mapping  # {'person': 0, 'car': 1}
        self.class_names = {v: k for k, v in class_mapping.items()}  # {0: 'person', 1: 'car'}
        self.min_samples = min_samples
        self.buffer = []
        self.is_training = False
        self.last_trained_at = None
        
        # Training progress tracking
        self.current_epoch = 0
        self.total_epochs = 50
        self.current_loss = 0.0
        
        print(f"[Shadow] Initialized with {len(class_mapping)} classes: {list(class_mapping.keys())}")

    def add_labels(self, samples: list) -> dict:
        """
        Add samples to training buffer.
        
        Args:
            samples: List of sample dicts with required keys:
                - image_path
                - detections (with class_id)
                - width, height
                - entropy
        """
        valid_samples = 0
        for s in samples:
            # Validate sample structure
            if not self._validate_sample(s):
                print(f"[Shadow] Invalid sample: {s.get('image_path', 'unknown')}")
                continue
            
            self.buffer.append(s)
            valid_samples += 1
        
        ready = len(self.buffer) >= self.min_samples
        
        return {
            'buffer_size': len(self.buffer),
            'ready_to_train': ready,
            'is_training': self.is_training,
            'valid_samples_added': valid_samples
        }

    def _validate_sample(self, sample: dict) -> bool:
        """Validate that sample has all required fields."""
        required = ['image_path', 'detections', 'width', 'height']
        
        for field in required:
            if field not in sample:
                print(f"[Shadow] Missing field: {field}")
                return False
        
        # Validate dimensions
        if sample['width'] <= 0 or sample['height'] <= 0:
            print(f"[Shadow] Invalid dimensions: {sample['width']}x{sample['height']}")
            return False
        
        # Validate detections have class_id
        for det in sample['detections']:
            if 'class_id' not in det:
                print(f"[Shadow] Detection missing class_id")
                return False
        
        return True

    def train(self, replay_samples: list = None) -> dict:
        """
        Train shadow model on buffered samples + replay samples.
        
        Args:
            replay_samples: Optional list of samples from replay buffer
            
        Returns:
            dict: Training result with success status and metrics
        """
        if self.is_training:
            return {'success': False, 'error': 'Already training'}
        
        if len(self.buffer) < self.min_samples:
            return {
                'success': False, 
                'error': f'Not enough samples: {len(self.buffer)}/{self.min_samples}'
            }

        self.is_training = True
        
        try:
            # Combine new samples with replay samples
            all_samples = self.buffer + (replay_samples or [])
            print(f"[Shadow] Training on {len(self.buffer)} new + {len(replay_samples or [])} replay samples")
            
            # Validate all samples
            valid_samples = [s for s in all_samples if self._validate_sample(s)]
            if len(valid_samples) < self.min_samples:
                return {
                    'success': False,
                    'error': f'Only {len(valid_samples)} valid samples after validation'
                }
            
            # Create temporary dataset
            tmp_dir = Path(tempfile.mkdtemp())
            dataset_yaml = self._create_yolo_dataset(valid_samples, tmp_dir)
            
            print(f"[Shadow] Starting training on {len(valid_samples)} samples...")
            print(f"[Shadow] Dataset created at: {tmp_dir}")
            
            # Load model from base (or could use active model for continual learning)
            model = YOLO(self.base_model_path)
            
            # 🔒 CRITICAL: Freeze backbone to prevent catastrophic forgetting
            self._freeze_backbone(model, freeze_layers=10)
            
            # Setup training callbacks
            def on_train_epoch_end(trainer):
                """Callback to track training progress."""
                self.current_epoch = trainer.epoch
                metrics = trainer.metrics
                if hasattr(metrics, 'box_loss'):
                    self.current_loss = float(metrics.box_loss)
                print(f"[Shadow] Epoch {self.current_epoch}/{self.total_epochs}, Loss: {self.current_loss:.4f}")
            
            # Train with frozen backbone
            results = model.train(
                data=str(dataset_yaml),
                epochs=self.total_epochs,
                imgsz=640,
                batch=8,
                patience=10,
                device=0,  # GPU index (use 'cpu' for CPU training)
                verbose=False,
                project=str(tmp_dir),
                name="shadow_run",
                exist_ok=True,
                # Callbacks
                callbacks={'on_train_epoch_end': on_train_epoch_end}
            )
            
            # Locate best weights
            train_dir = tmp_dir / "shadow_run"
            best_weight = self._find_best_weights(train_dir)
            
            if not best_weight:
                raise Exception("No valid weights generated during training")
            
            # Move result to permanent location
            save_path = Path("models/shadow_candidate.pt")
            save_path.parent.mkdir(exist_ok=True)
            shutil.copy2(best_weight, save_path)
            
            print(f"[Shadow] Training complete! Model saved to: {save_path}")
            
            # Get final metrics
            final_metrics = {
                'map50': getattr(results, 'map50', 0.0),
                'map': getattr(results, 'map', 0.0),
                'final_loss': self.current_loss
            }
            
            # Cleanup
            trained_paths = [s['image_path'] for s in self.buffer]
            self.buffer.clear()
            self.last_trained_at = datetime.now().isoformat()
            
            # Clean up temp directory
            try:
                shutil.rmtree(tmp_dir)
            except:
                print(f"[Shadow] Warning: Could not remove temp dir {tmp_dir}")
            
            return {
                'success': True, 
                'save_path': str(save_path),
                'sample_count': len(valid_samples),
                'new_samples': len(self.buffer),
                'replay_samples': len(replay_samples or []),
                'timestamp': self.last_trained_at,
                'trained_paths': trained_paths,
                'metrics': final_metrics
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[Shadow] Training error: {e}")
            print(error_details)
            return {
                'success': False, 
                'error': str(e),
                'details': error_details
            }
        finally:
            self.is_training = False
            self.current_epoch = 0
            self.current_loss = 0.0

    def _freeze_backbone(self, model, freeze_layers=10):
        """
        Freeze early layers to prevent catastrophic forgetting.
        
        Args:
            model: YOLO model instance
            freeze_layers: Number of layers to freeze (default: 10)
        """
        try:
            frozen_count = 0
            for i, (name, param) in enumerate(model.model.named_parameters()):
                if i < freeze_layers:
                    param.requires_grad = False
                    frozen_count += 1
            
            print(f"[Shadow] Frozen {frozen_count} parameters in first {freeze_layers} layers")
        except Exception as e:
            print(f"[Shadow] Warning: Could not freeze backbone: {e}")
            print("[Shadow] Continuing without frozen layers (may cause catastrophic forgetting)")

    def _find_best_weights(self, train_dir: Path) -> Path:
        """
        Locate best model weights from training run.
        
        Priority: best.pt > last.pt > any .pt file
        """
        weights_dir = train_dir / "weights"
        
        # Check for best.pt (preferred)
        if (weights_dir / "best.pt").exists():
            print("[Shadow] Using best.pt")
            return weights_dir / "best.pt"
        
        # Check for last.pt (fallback)
        if (weights_dir / "last.pt").exists():
            print("[Shadow] Using last.pt (best.pt not found)")
            return weights_dir / "last.pt"
        
        # Find any .pt file
        weights = list(train_dir.glob("**/*.pt"))
        if weights:
            print(f"[Shadow] Using {weights[0].name} (no standard checkpoints found)")
            return weights[0]
        
        return None

    def _create_yolo_dataset(self, samples: list, root_dir: Path) -> Path:
        """
        Generate YOLO-format dataset from samples.
        
        Structure:
        root/
          data.yaml
          train/
            images/
            labels/
        
        Args:
            samples: List of validated sample dicts
            root_dir: Root directory for dataset
            
        Returns:
            Path to data.yaml file
        """
        images_dir = root_dir / "train" / "images"
        labels_dir = root_dir / "train" / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        skipped_count = 0
        
        for s in samples:
            src_img = Path(s['image_path'])
            if not src_img.exists():
                print(f"[Shadow] Image not found: {src_img}")
                skipped_count += 1
                continue
            
            # Copy image
            dst_img = images_dir / src_img.name
            shutil.copy2(src_img, dst_img)
            
            # Create label file
            label_file = labels_dir / f"{src_img.stem}.txt"
            
            # Get image dimensions
            w_img = s['width']
            h_img = s['height']
            
            with open(label_file, "w") as f:
                for det in s['detections']:
                    # Get class ID (should already be in detection)
                    cls_id = det.get('class_id')
                    if cls_id is None:
                        print(f"[Shadow] Detection missing class_id, skipping")
                        continue
                    
                    # Get bbox in xyxy format
                    bbox = det['bbox']
                    
                    # Convert xyxy to xywhn (YOLO format)
                    x_c = ((bbox[0] + bbox[2]) / 2) / w_img
                    y_c = ((bbox[1] + bbox[3]) / 2) / h_img
                    w = (bbox[2] - bbox[0]) / w_img
                    h = (bbox[3] - bbox[1]) / h_img
                    
                    # Clamp to [0, 1]
                    x_c = max(0.0, min(1.0, x_c))
                    y_c = max(0.0, min(1.0, y_c))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))
                    
                    # Write YOLO format: class x_center y_center width height
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            copied_count += 1

        print(f"[Shadow] Dataset created: {copied_count} images, {skipped_count} skipped")
        
        # Create data.yaml
        yaml_path = root_dir / "data.yaml"
        yaml_data = {
            'path': str(root_dir.absolute()),
            'train': 'train/images',
            'val': 'train/images',  # Using train as val for shadow training (fast iteration)
            'names': self.class_names  # {0: 'person', 1: 'car', ...}
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)
        
        print(f"[Shadow] data.yaml created with {len(self.class_names)} classes")
        
        return yaml_path

    def get_status(self) -> dict:
        """Get current training status."""
        return {
            'is_training': self.is_training,
            'buffer_size': len(self.buffer),
            'last_trained': self.last_trained_at,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_loss': self.current_loss,
            'progress_percent': (self.current_epoch / self.total_epochs * 100) if self.total_epochs > 0 else 0
        }

    def get_training_progress(self) -> dict:
        """Get detailed training progress (for UI updates)."""
        if not self.is_training:
            return {
                'training': False,
                'ready': len(self.buffer) >= self.min_samples,
                'buffer_size': len(self.buffer),
                'min_samples': self.min_samples
            }
        
        return {
            'training': True,
            'epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'loss': round(self.current_loss, 4),
            'percent': round((self.current_epoch / self.total_epochs) * 100, 1) if self.total_epochs > 0 else 0
        }

    def clear_buffer(self):
        """Clear training buffer (for manual reset)."""
        size = len(self.buffer)
        self.buffer.clear()
        return {'cleared': size}

    def ping(self) -> str:
        """Health check for Ray actor."""
        return 'pong'