# src/core/training_orchestrator.py
"""
Training Orchestrator - Manages background training lifecycle
Handles Ray initialization, training triggers, and model promotion
"""

import ray
from pathlib import Path
from typing import Optional, Dict, List, Callable
from datetime import datetime


class TrainingOrchestrator:
    """Manages shadow model training workflow."""
    
    def __init__(
        self,
        data_manager,
        model_manager,
        replay_buffer,
        min_samples: int = 30,
        num_gpus: int = 1
    ):
        """
        Args:
            data_manager: DataManager instance for label storage
            model_manager: ModelManager instance for model versioning
            replay_buffer: ReplayBuffer instance for experience replay
            min_samples: Minimum samples needed to trigger training
            num_gpus: Number of GPUs to allocate to Ray
        """
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.replay_buffer = replay_buffer
        self.min_samples = min_samples
        self.num_gpus = num_gpus
        
        # Training state
        self.shadow_trainer = None
        self.training_future = None
        self.ray_initialized = False
        
        # Callbacks for UI updates
        self.on_status_change: Optional[Callable] = None
        self.on_training_complete: Optional[Callable] = None
        self.on_training_failed: Optional[Callable] = None
        
        # Initialize Ray
        self._init_ray()
    
    def _init_ray(self) -> bool:
        """Initialize Ray for distributed training."""
        try:
            if ray.is_initialized():
                print("[Orchestrator] Ray already initialized")
                self.ray_initialized = True
                return True
            
            ray.init(
                num_gpus=self.num_gpus,
                ignore_reinit_error=True,
                logging_level="ERROR"
            )
            print("[Orchestrator] Ray initialized successfully")
            self.ray_initialized = True
            
            # Try to create shadow trainer if we have class mapping
            self._try_create_trainer()
            return True
            
        except ImportError:
            print("[Orchestrator] Ray not installed - background training disabled")
            self.ray_initialized = False
            return False
        except Exception as e:
            print(f"[Orchestrator] Ray init failed: {e}")
            self.ray_initialized = False
            return False
    
    def _try_create_trainer(self):
        """Attempt to create shadow trainer if class mapping exists."""
        if not self.ray_initialized:
            return False
        
        class_mapping = self.data_manager.data.get('class_mapping', {})
        if not class_mapping:
            print("[Orchestrator] No class mapping yet - waiting for first label")
            return False
        
        try:
            from src.core.shadow_trainer import ShadowTrainer
            
            base_model = self.model_manager.resolve_active_path()
            self.shadow_trainer = ShadowTrainer.remote(
                base_model_path=base_model,
                class_mapping=class_mapping,
                min_samples=self.min_samples
            )
            print(f"[Orchestrator] Shadow trainer created with {len(class_mapping)} classes")
            return True
            
        except Exception as e:
            print(f"[Orchestrator] Failed to create shadow trainer: {e}")
            return False
    
    def check_training_trigger(self) -> bool:
        """
        Check if training should be triggered based on queue size.
        
        Returns:
            True if training was triggered, False otherwise
        """
        if not self.ray_initialized:
            return False
        
        # Ensure trainer exists
        if not self.shadow_trainer:
            self._try_create_trainer()
            if not self.shadow_trainer:
                return False
        
        # Check if already training
        if self.is_training():
            print("[Orchestrator] Training already in progress")
            return False
        
        # Check queue size
        stats = self.data_manager.get_stats()
        queue_size = stats['training_queue_size']
        
        if queue_size >= self.min_samples:
            print(f"[Orchestrator] Queue full ({queue_size}/{self.min_samples}) - triggering training")
            return self.trigger_training()
        
        return False
    
    def is_training(self) -> bool:
        """Check if training is currently in progress."""
        if not self.training_future:
            return False
        
        try:
            ready, _ = ray.wait([self.training_future], timeout=0)
            return len(ready) == 0  # Not ready = still training
        except:
            return False
    
    def trigger_training(self) -> bool:
        """
        Manually trigger shadow model training.
        
        Returns:
            True if training started successfully, False otherwise
        """
        if not self.shadow_trainer:
            print("[Orchestrator] Shadow trainer not available")
            return False
        
        if self.is_training():
            print("[Orchestrator] Training already in progress")
            return False
        
        try:
            # Get training samples
            samples = self.data_manager.get_training_batch(
                count=self.min_samples,
                new_only=True,
                return_full_samples=True
            )
            
            if len(samples) < self.min_samples:
                print(f"[Orchestrator] Not enough samples: {len(samples)}/{self.min_samples}")
                return False
            
            # Get replay samples
            replay_paths = self.data_manager.get_replay_samples(count=10)
            replay_samples = self.data_manager.prepare_training_samples(replay_paths)
            
            # Add current samples to training batch
            all_samples = samples + replay_samples
            
            # Start training
            print(f"[Orchestrator] Starting training: {len(samples)} new + {len(replay_samples)} replay")
            self.training_future = self.shadow_trainer.train.remote(all_samples)
            
            # Notify UI
            if self.on_status_change:
                self.on_status_change({
                    'status': 'training_started',
                    'sample_count': len(samples),
                    'replay_count': len(replay_samples)
                })
            
            return True
            
        except Exception as e:
            print(f"[Orchestrator] Error triggering training: {e}")
            if self.on_training_failed:
                self.on_training_failed({'error': str(e)})
            return False
    
    def get_training_status(self) -> Dict:
        """
        Get current training status.
        
        Returns:
            Dictionary with training progress information
        """
        if not self.shadow_trainer:
            return {
                'available': False,
                'training': False,
                'reason': 'trainer_not_initialized'
            }
        
        try:
            status_future = self.shadow_trainer.get_training_progress.remote()
            status = ray.get(status_future, timeout=1)
            status['available'] = True
            return status
            
        except ray.exceptions.GetTimeoutError:
            return {
                'available': True,
                'training': False,
                'reason': 'status_timeout'
            }
        except Exception as e:
            return {
                'available': False,
                'training': False,
                'error': str(e)
            }
    
    def check_training_completion(self) -> Optional[Dict]:
        """
        Check if training has completed and handle result.
        
        Returns:
            Training result dict if completed, None if still training
        """
        if not self.training_future:
            return None
        
        try:
            ready, _ = ray.wait([self.training_future], timeout=0)
            
            if not ready:
                return None  # Still training
            
            # Training completed - get result
            result = ray.get(self.training_future)
            self.training_future = None
            
            if result['success']:
                self._handle_training_success(result)
            else:
                self._handle_training_failure(result)
            
            return result
            
        except Exception as e:
            print(f"[Orchestrator] Error checking completion: {e}")
            return None
    
    def _handle_training_success(self, result: Dict):
        """Handle successful training completion."""
        print(f"[Orchestrator] Training completed successfully!")
        print(f"[Orchestrator] Trained on {result['sample_count']} samples")
        print(f"[Orchestrator] Model saved to: {result['save_path']}")
        
        # Mark samples as trained
        trained_paths = result.get('trained_paths', [])
        self.data_manager.mark_trained(trained_paths)
        
        # Add to replay buffer
        replay_samples = self.data_manager.prepare_training_samples(trained_paths)
        self.replay_buffer.add(replay_samples)
        
        # Notify UI
        if self.on_training_complete:
            self.on_training_complete(result)
    
    def _handle_training_failure(self, result: Dict):
        """Handle training failure."""
        error = result.get('error', 'Unknown error')
        print(f"[Orchestrator] Training failed: {error}")
        
        # Notify UI
        if self.on_training_failed:
            self.on_training_failed(result)
    
    def promote_shadow_model(self, validate: bool = True) -> Dict:
        """
        Promote shadow model to active.
        
        Args:
            validate: Whether to validate model before promotion
        
        Returns:
            Result dictionary with success status and details
        """
        shadow_path = "models/shadow_candidate.pt"
        
        if not Path(shadow_path).exists():
            return {
                'success': False,
                'error': 'No shadow model found. Train a model first.'
            }
        
        # Optional validation
        if validate:
            comparison = self.model_manager.compare_models(
                base_path=self.model_manager.resolve_active_path(),
                shadow_path=shadow_path
            )
            
            if not comparison.get('recommend_promote', False):
                return {
                    'success': False,
                    'error': comparison.get('error', 'Model validation failed'),
                    'requires_confirmation': True,
                    'comparison': comparison
                }
        
        # Perform promotion
        result = self.model_manager.promote_shadow(shadow_path, validate=validate)
        
        if result['success']:
            print(f"[Orchestrator] Shadow model promoted: {result['version']}")
        
        return result
    
    def get_queue_status(self) -> Dict:
        """Get status of training queue."""
        stats = self.data_manager.get_stats()
        return {
            'queue_size': stats['training_queue_size'],
            'min_samples': self.min_samples,
            'ready_to_train': stats['training_queue_size'] >= self.min_samples,
            'progress_percent': min(100, (stats['training_queue_size'] / self.min_samples) * 100)
        }
    
    def shutdown(self):
        """Cleanup resources."""
        print("[Orchestrator] Shutting down...")
        
        # Ray will be shut down when the process exits
        # We don't explicitly call ray.shutdown() here to avoid conflicts
        
        self.shadow_trainer = None
        self.training_future = None
        print("[Orchestrator] Shutdown complete")