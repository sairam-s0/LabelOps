#!/usr/bin/env python3
"""
Smart Labeling System Test Suite
Automated validation of all components with zero user intervention
Tests CPU/GPU capabilities, Ray distributed training, and GUI functionality
"""

import sys
import subprocess
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import traceback

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.score = 0
        self.max_score = 100
        self.message = ""
        self.details = []
        self.warnings = []
        self.time_taken = 0
    
    def to_dict(self):
        return {
            'name': self.name,
            'passed': self.passed,
            'score': self.score,
            'max_score': self.max_score,
            'message': self.message,
            'details': self.details,
            'warnings': self.warnings,
            'time_taken': self.time_taken
        }

class SystemTester:
    def __init__(self, auto_install=True):
        self.auto_install = auto_install
        self.results = []
        self.total_score = 0
        self.max_total_score = 0
        self.has_gpu = False
        self.has_ray = False
        self.test_dir = None
        
    def print_header(self, text):
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
        print(f"  {text}")
        print(f"{'='*70}{Colors.ENDC}\n")
    
    def print_test(self, name):
        print(f"{Colors.OKBLUE}Testing: {name}...{Colors.ENDC}", end=" ", flush=True)
    
    def print_pass(self, message=""):
        print(f"{Colors.OKGREEN}✓ PASS{Colors.ENDC} {message}")
    
    def print_fail(self, message=""):
        print(f"{Colors.FAIL}✗ FAIL{Colors.ENDC} {message}")
    
    def print_warning(self, message=""):
        print(f"{Colors.WARNING}⚠ WARNING{Colors.ENDC} {message}")
    
    def run_all_tests(self):
        """Run complete test suite."""
        self.print_header("SMART LABELING SYSTEM TEST SUITE")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Platform: {sys.platform}")
        
        # Setup test directory
        self.test_dir = Path(tempfile.mkdtemp(prefix="smart_label_test_"))
        print(f"Test directory: {self.test_dir}\n")
        
        try:
            # Phase 1: Environment Setup
            self.print_header("PHASE 1: ENVIRONMENT SETUP")
            self.test_requirements_file()
            if self.auto_install:
                self.test_install_dependencies()
            self.test_python_version()
            
            # Phase 2: Core Dependencies
            self.print_header("PHASE 2: CORE DEPENDENCIES")
            self.test_import_numpy()
            self.test_import_pillow()
            self.test_import_pyqt6()
            self.test_import_ultralytics()
            
            # Phase 3: Hardware Detection
            self.print_header("PHASE 3: HARDWARE CAPABILITIES")
            self.test_gpu_availability()
            self.test_cuda_functionality()
            
            # Phase 4: Ray Framework
            self.print_header("PHASE 4: RAY DISTRIBUTED FRAMEWORK")
            self.test_import_ray()
            self.test_ray_cpu_mode()
            if self.has_gpu:
                self.test_ray_gpu_mode()
            self.test_ray_actor_creation()
            
            # Phase 5: YOLO Model
            self.print_header("PHASE 5: YOLO MODEL FUNCTIONALITY")
            self.test_yolo_model_download()
            self.test_yolo_inference()
            self.test_yolo_class_names()
            
            # Phase 6: Core Modules
            self.print_header("PHASE 6: CORE MODULES")
            self.test_data_manager()
            self.test_model_manager()
            self.test_replay_buffer()
            self.test_shadow_trainer()
            self.test_training_orchestrator()
            
            # Phase 7: GUI Components
            self.print_header("PHASE 7: GUI COMPONENTS")
            self.test_gui_initialization()
            self.test_gui_dialogs()
            
            # Phase 8: Integration Tests
            self.print_header("PHASE 8: INTEGRATION TESTS")
            self.test_end_to_end_labeling()
            self.test_training_workflow()
            
            # Generate report
            self.generate_report()
            
        finally:
            # Cleanup
            self.cleanup()
    
    def test_requirements_file(self):
        """Check if requirements.txt exists."""
        result = TestResult("Requirements File")
        self.print_test("Requirements file existence")
        
        try:
            req_file = Path("requirements.txt")
            if req_file.exists():
                with open(req_file) as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                result.passed = True
                result.score = 100
                result.message = f"Found {len(lines)} dependencies"
                result.details = lines[:5]  # First 5 packages
                self.print_pass(result.message)
            else:
                result.message = "requirements.txt not found"
                self.print_fail(result.message)
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_install_dependencies(self):
        """Auto-install dependencies from requirements.txt."""
        result = TestResult("Dependency Installation")
        self.print_test("Installing dependencies")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if proc.returncode == 0:
                result.passed = True
                result.score = 100
                result.message = "All dependencies installed successfully"
                self.print_pass()
            else:
                result.message = f"Installation failed: {proc.stderr[:200]}"
                self.print_fail(result.message)
        except subprocess.TimeoutExpired:
            result.message = "Installation timeout (>5 minutes)"
            self.print_fail(result.message)
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_python_version(self):
        """Check Python version compatibility."""
        result = TestResult("Python Version")
        self.print_test("Python version compatibility")
        
        try:
            major, minor = sys.version_info[:2]
            if major == 3 and minor >= 8:
                result.passed = True
                result.score = 100
                result.message = f"Python {major}.{minor} (✓ Compatible)"
                self.print_pass(result.message)
            else:
                result.message = f"Python {major}.{minor} (Requires 3.8+)"
                self.print_fail(result.message)
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_import_numpy(self):
        """Test NumPy import and functionality."""
        result = TestResult("NumPy")
        self.print_test("NumPy import and array operations")
        
        try:
            import numpy as np
            arr = np.array([1, 2, 3, 4, 5])
            assert arr.mean() == 3.0
            result.passed = True
            result.score = 100
            result.message = f"NumPy {np.__version__}"
            self.print_pass(result.message)
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_import_pillow(self):
        """Test Pillow (PIL) import and image operations."""
        result = TestResult("Pillow (PIL)")
        self.print_test("Pillow import and image operations")
        
        try:
            from PIL import Image, ImageDraw
            # Create test image
            img = Image.new('RGB', (100, 100), color='red')
            draw = ImageDraw.Draw(img)
            draw.rectangle([10, 10, 90, 90], outline='blue', width=2)
            assert img.size == (100, 100)
            img.close()
            result.passed = True
            result.score = 100
            result.message = f"Pillow {Image.__version__}"
            self.print_pass(result.message)
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_import_pyqt6(self):
        """Test PyQt6 import (without creating GUI)."""
        result = TestResult("PyQt6")
        self.print_test("PyQt6 import (GUI framework)")
        
        try:
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import Qt, QTimer
            from PyQt6.QtGui import QPixmap
            result.passed = True
            result.score = 100
            result.message = "PyQt6 available"
            self.print_pass(result.message)
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_import_ultralytics(self):
        """Test Ultralytics YOLO import."""
        result = TestResult("Ultralytics")
        self.print_test("Ultralytics YOLO import")
        
        try:
            from ultralytics import YOLO
            import ultralytics
            result.passed = True
            result.score = 100
            result.message = f"Ultralytics {ultralytics.__version__}"
            self.print_pass(result.message)
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_gpu_availability(self):
        """Detect GPU availability."""
        result = TestResult("GPU Detection")
        self.print_test("GPU/CUDA availability")
        
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            
            if self.has_gpu:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                result.passed = True
                result.score = 100
                result.message = f"GPU: {gpu_name} ({gpu_count} device(s))"
                result.details.append(f"CUDA Version: {torch.version.cuda}")
                self.print_pass(result.message)
            else:
                result.passed = True
                result.score = 50
                result.message = "No GPU detected (CPU mode will be used)"
                result.warnings.append("Training will be slower on CPU")
                self.print_warning(result.message)
        except ImportError:
            result.passed = True
            result.score = 50
            result.message = "PyTorch not installed (cannot detect GPU)"
            self.print_warning(result.message)
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_cuda_functionality(self):
        """Test CUDA tensor operations if GPU available."""
        result = TestResult("CUDA Operations")
        self.print_test("CUDA tensor operations")
        
        if not self.has_gpu:
            result.passed = True
            result.score = 100
            result.message = "Skipped (no GPU)"
            self.print_warning("Skipped - no GPU")
            self.results.append(result)
            return
        
        try:
            import torch
            # Create tensor on GPU
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            assert z.is_cuda
            result.passed = True
            result.score = 100
            result.message = "CUDA operations successful"
            self.print_pass()
        except Exception as e:
            result.message = f"CUDA operations failed: {str(e)}"
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_import_ray(self):
        """Test Ray import."""
        result = TestResult("Ray Framework")
        self.print_test("Ray distributed framework import")
        
        try:
            import ray
            self.has_ray = True
            result.passed = True
            result.score = 100
            result.message = f"Ray {ray.__version__}"
            self.print_pass(result.message)
        except ImportError:
            result.message = "Ray not installed (background training unavailable)"
            result.warnings.append("Install with: pip install ray")
            self.print_fail(result.message)
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_ray_cpu_mode(self):
        """Test Ray initialization in CPU mode."""
        result = TestResult("Ray CPU Mode")
        self.print_test("Ray initialization (CPU mode)")
        
        if not self.has_ray:
            result.message = "Skipped (Ray not available)"
            self.print_warning("Skipped")
            self.results.append(result)
            return
        
        try:
            import ray
            
            # Initialize Ray with CPU only
            if ray.is_initialized():
                ray.shutdown()
            
            ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True)
            
            # Test simple task
            @ray.remote
            def test_task(x):
                return x * 2
            
            future = test_task.remote(5)
            result_val = ray.get(future, timeout=5)
            assert result_val == 10
            
            ray.shutdown()
            
            result.passed = True
            result.score = 100
            result.message = "Ray CPU mode working"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_ray_gpu_mode(self):
        """Test Ray initialization in GPU mode."""
        result = TestResult("Ray GPU Mode")
        self.print_test("Ray initialization (GPU mode)")
        
        if not self.has_ray:
            result.message = "Skipped (Ray not available)"
            self.print_warning("Skipped")
            self.results.append(result)
            return
        
        if not self.has_gpu:
            result.message = "Skipped (no GPU)"
            self.print_warning("Skipped")
            self.results.append(result)
            return
        
        try:
            import ray
            
            if ray.is_initialized():
                ray.shutdown()
            
            ray.init(num_gpus=1, ignore_reinit_error=True)
            
            # Test GPU task
            @ray.remote(num_gpus=1)
            def gpu_task():
                import torch
                return torch.cuda.is_available()
            
            future = gpu_task.remote()
            gpu_available = ray.get(future, timeout=10)
            assert gpu_available
            
            ray.shutdown()
            
            result.passed = True
            result.score = 100
            result.message = "Ray GPU mode working"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            result.warnings.append("GPU mode failed, but CPU mode available")
            self.print_warning(result.message)
        
        self.results.append(result)
    
    def test_ray_actor_creation(self):
        """Test Ray actor creation and method calls."""
        result = TestResult("Ray Actors")
        self.print_test("Ray actor creation and communication")
        
        if not self.has_ray:
            result.message = "Skipped (Ray not available)"
            self.print_warning("Skipped")
            self.results.append(result)
            return
        
        try:
            import ray
            
            if ray.is_initialized():
                ray.shutdown()
            
            ray.init(num_cpus=2, ignore_reinit_error=True)
            
            @ray.remote
            class TestActor:
                def __init__(self):
                    self.value = 0
                
                def increment(self):
                    self.value += 1
                    return self.value
            
            actor = TestActor.remote()
            result1 = ray.get(actor.increment.remote())
            result2 = ray.get(actor.increment.remote())
            assert result1 == 1 and result2 == 2
            
            ray.shutdown()
            
            result.passed = True
            result.score = 100
            result.message = "Ray actors working correctly"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_yolo_model_download(self):
        """Test YOLO model download."""
        result = TestResult("YOLO Model Download")
        self.print_test("YOLO model download and loading")
        
        try:
            from ultralytics import YOLO
            
            # This will download model if not present
            model = YOLO('yolov8n.pt')  # Use nano for faster testing
            
            result.passed = True
            result.score = 100
            result.message = "YOLO model loaded successfully"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_yolo_inference(self):
        """Test YOLO inference on dummy image."""
        result = TestResult("YOLO Inference")
        self.print_test("YOLO inference on test image")
        
        try:
            from ultralytics import YOLO
            from PIL import Image
            import numpy as np
            
            # Create test image
            img = Image.new('RGB', (640, 480), color='blue')
            
            # Load model and run inference
            model = YOLO('yolov8n.pt')
            results = model(img, verbose=False)
            
            assert len(results) > 0
            img.close()
            
            result.passed = True
            result.score = 100
            result.message = "YOLO inference successful"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_yolo_class_names(self):
        """Test YOLO class name extraction."""
        result = TestResult("YOLO Class Names")
        self.print_test("YOLO class name extraction")
        
        try:
            from ultralytics import YOLO
            
            model = YOLO('yolov8n.pt')
            class_names = list(model.names.values())
            
            assert len(class_names) >= 80  # COCO has 80 classes
            assert 'person' in class_names
            assert 'car' in class_names
            
            result.passed = True
            result.score = 100
            result.message = f"Found {len(class_names)} classes"
            result.details = class_names[:5]  # First 5 classes
            self.print_pass(result.message)
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_data_manager(self):
        """Test DataManager functionality."""
        result = TestResult("DataManager")
        self.print_test("DataManager (label storage)")
        
        try:
            # Add src to path
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from core.data_manager import DataManager
            
            test_file = self.test_dir / "test_labels.json"
            dm = DataManager(str(test_file))
            
            # Test save labels
            success = dm.save_labels(
                image_path="test.jpg",
                detections=[{'bbox': [0, 0, 100, 100], 'class': 'person', 'confidence': 0.95}],
                entropy=0.5,
                img_width=640,
                img_height=480
            )
            
            assert success
            assert test_file.exists()
            
            # Test get stats
            stats = dm.get_stats()
            assert stats['total_labeled'] == 1
            
            result.passed = True
            result.score = 100
            result.message = "DataManager working correctly"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            result.details.append(traceback.format_exc())
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_model_manager(self):
        """Test ModelManager functionality."""
        result = TestResult("ModelManager")
        self.print_test("ModelManager (version control)")
        
        try:
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from core.model_manager import ModelManager
            
            test_models = self.test_dir / "models"
            mm = ModelManager(str(test_models))
            
            # Test active model resolution
            active = mm.resolve_active_path()
            assert active is not None
            
            result.passed = True
            result.score = 100
            result.message = "ModelManager working correctly"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            result.details.append(traceback.format_exc())
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_replay_buffer(self):
        """Test ReplayBuffer functionality."""
        result = TestResult("ReplayBuffer")
        self.print_test("ReplayBuffer (experience replay)")
        
        try:
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from core.replay_buffer import ReplayBuffer
            
            rb = ReplayBuffer(max_size=10)
            
            # Add samples
            for i in range(5):
                rb.add({'id': i, 'data': f'sample_{i}'})
            
            assert len(rb) == 5
            
            # Test sampling
            samples = rb.sample(3)
            assert len(samples) == 3
            
            result.passed = True
            result.score = 100
            result.message = "ReplayBuffer working correctly"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            result.details.append(traceback.format_exc())
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_shadow_trainer(self):
        """Test ShadowTrainer actor creation."""
        result = TestResult("ShadowTrainer")
        self.print_test("ShadowTrainer (background training)")
        
        if not self.has_ray:
            result.message = "Skipped (Ray not available)"
            self.print_warning("Skipped")
            self.results.append(result)
            return
        
        try:
            import ray
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from core.shadow_trainer import ShadowTrainer
            
            if ray.is_initialized():
                ray.shutdown()
            
            ray.init(num_cpus=2, ignore_reinit_error=True)
            
            # Create actor
            trainer = ShadowTrainer.remote(
                base_model_path='yolov8n.pt',
                class_mapping={'person': 0, 'car': 1},
                min_samples=5
            )
            
            # Test status
            status = ray.get(trainer.get_training_progress.remote(), timeout=5)
            assert 'training' in status
            
            ray.shutdown()
            
            result.passed = True
            result.score = 100
            result.message = "ShadowTrainer actor created successfully"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            result.details.append(traceback.format_exc())
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_training_orchestrator(self):
        """Test TrainingOrchestrator."""
        result = TestResult("TrainingOrchestrator")
        self.print_test("TrainingOrchestrator (workflow manager)")
        
        try:
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from core.training_orchestrator import TrainingOrchestrator
            from core.data_manager import DataManager
            from core.model_manager import ModelManager
            from core.replay_buffer import ReplayBuffer
            
            dm = DataManager(str(self.test_dir / "labels.json"))
            mm = ModelManager(str(self.test_dir / "models"))
            rb = ReplayBuffer(max_size=10)
            
            orchestrator = TrainingOrchestrator(
                data_manager=dm,
                model_manager=mm,
                replay_buffer=rb,
                min_samples=5,
                num_gpus=1 if self.has_gpu else 0
            )
            
            # Test queue status
            status = orchestrator.get_queue_status()
            assert 'queue_size' in status
            
            result.passed = True
            result.score = 100
            result.message = "TrainingOrchestrator initialized successfully"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            result.details.append(traceback.format_exc())
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_gui_initialization(self):
        """Test GUI initialization without display."""
        result = TestResult("GUI Initialization")
        self.print_test("GUI initialization (no display)")
        
        try:
            # Set headless mode for testing
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            
            from PyQt6.QtWidgets import QApplication
            
            # Create application without showing window
            app = QApplication.instance() or QApplication(sys.argv)
            
            result.passed = True
            result.score = 100
            result.message = "GUI components can initialize"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            result.warnings.append("GUI may not work in headless environments")
            self.print_warning(result.message)
        
        self.results.append(result)
    
    def test_gui_dialogs(self):
        """Test dialog creation."""
        result = TestResult("GUI Dialogs")
        self.print_test("GUI dialog components")
        
        try:
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            
            # Import main to test ClassSelectorDialog
            import main
            
            result.passed = True
            result.score = 100
            result.message = "Dialog components available"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_end_to_end_labeling(self):
        """Test complete labeling workflow."""
        result = TestResult("E2E Labeling Workflow")
        self.print_test("End-to-end labeling workflow")
        
        try:
            from PIL import Image
            import json
            
            # Create test images
            test_images = self.test_dir / "images"
            test_images.mkdir(exist_ok=True)
            
            for i in range(3):
                img = Image.new('RGB', (640, 480), color='red')
                img.save(test_images / f"test_{i}.jpg")
                img.close()
            
            # Simulate labeling workflow
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from core.data_manager import DataManager
            
            dm = DataManager(str(self.test_dir / "workflow_labels.json"))
            
            for i in range(3):
                dm.save_labels(
                    image_path=str(test_images / f"test_{i}.jpg"),
                    detections=[{'bbox': [10, 10, 100, 100], 'class': 'person', 'confidence': 0.9}],
                    entropy=0.5,
                    img_width=640,
                    img_height=480
                )
            
            stats = dm.get_stats()
            assert stats['total_labeled'] == 3
            
            result.passed = True
            result.score = 100
            result.message = "Labeling workflow completed successfully"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            result.details.append(traceback.format_exc())
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def test_training_workflow(self):
        """Test training trigger and workflow."""
        result = TestResult("Training Workflow")
        self.print_test("Training workflow simulation")
        
        if not self.has_ray:
            result.message = "Skipped (Ray not available)"
            self.print_warning("Skipped")
            self.results.append(result)
            return
        
        try:
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from core.training_orchestrator import TrainingOrchestrator
            from core.data_manager import DataManager
            from core.model_manager import ModelManager
            from core.replay_buffer import ReplayBuffer
            
            # Setup
            dm = DataManager(str(self.test_dir / "training_labels.json"))
            mm = ModelManager(str(self.test_dir / "models"))
            rb = ReplayBuffer(max_size=10)
            
            # Add enough samples to trigger training
            for i in range(10):
                dm.save_labels(
                    image_path=f"test_{i}.jpg",
                    detections=[{'bbox': [0, 0, 100, 100], 'class': 'person', 'confidence': 0.9}],
                    entropy=0.5,
                    img_width=640,
                    img_height=480
                )
            
            orchestrator = TrainingOrchestrator(
                data_manager=dm,
                model_manager=mm,
                replay_buffer=rb,
                min_samples=5,
                num_gpus=0  # CPU mode for testing
            )
            
            # Check queue status
            queue = orchestrator.get_queue_status()
            assert queue['queue_size'] >= 5
            
            result.passed = True
            result.score = 100
            result.message = "Training workflow configured correctly"
            self.print_pass()
        except Exception as e:
            result.message = str(e)
            result.details.append(traceback.format_exc())
            self.print_fail(result.message)
        
        self.results.append(result)
    
    def cleanup(self):
        """Cleanup test resources."""
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except:
            pass
        
        try:
            if self.test_dir and self.test_dir.exists():
                shutil.rmtree(self.test_dir)
        except:
            pass
    
    def generate_report(self):
        """Generate comprehensive test report."""
        self.print_header("TEST RESULTS SUMMARY")
        
        # Calculate scores
        for r in self.results:
            self.total_score += r.score
            self.max_total_score += r.max_score
        
        # Overall grade
        percentage = (self.total_score / self.max_total_score * 100) if self.max_total_score > 0 else 0
        
        if percentage >= 90:
            grade = "A"
            grade_color = Colors.OKGREEN
        elif percentage >= 75:
            grade = "B"
            grade_color = Colors.OKBLUE
        elif percentage >= 60:
            grade = "C"
            grade_color = Colors.WARNING
        else:
            grade = "F"
            grade_color = Colors.FAIL
        
        # Print summary
        print(f"\n{Colors.BOLD}Overall Score: {self.total_score}/{self.max_total_score} ({percentage:.1f}%)")
        print(f"Grade: {grade_color}{grade}{Colors.ENDC}\n")
        
        # System capabilities
        print(f"{Colors.BOLD}System Capabilities:{Colors.ENDC}")
        print(f"  • GPU Available: {Colors.OKGREEN + 'YES' if self.has_gpu else Colors.FAIL + 'NO'}{Colors.ENDC}")
        print(f"  • Ray Framework: {Colors.OKGREEN + 'YES' if self.has_ray else Colors.FAIL + 'NO'}{Colors.ENDC}")
        print(f"  • Recommended Mode: {Colors.OKGREEN + 'GPU Training' if self.has_gpu else Colors.WARNING + 'CPU Training'}{Colors.ENDC}")
        
        # Test breakdown
        print(f"\n{Colors.BOLD}Test Breakdown:{Colors.ENDC}")
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        print(f"  • Passed: {Colors.OKGREEN}{passed}{Colors.ENDC}")
        print(f"  • Failed: {Colors.FAIL}{failed}{Colors.ENDC}")
        
        # Failed tests detail
        if failed > 0:
            print(f"\n{Colors.FAIL}{Colors.BOLD}Failed Tests:{Colors.ENDC}")
            for r in self.results:
                if not r.passed:
                    print(f"  ✗ {r.name}: {r.message}")
        
        # Warnings
        warnings = [w for r in self.results for w in r.warnings]
        if warnings:
            print(f"\n{Colors.WARNING}{Colors.BOLD}Warnings:{Colors.ENDC}")
            for w in warnings:
                print(f"  ⚠ {w}")
        
        # Save JSON report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_score': self.total_score,
            'max_score': self.max_total_score,
            'percentage': percentage,
            'grade': grade,
            'has_gpu': self.has_gpu,
            'has_ray': self.has_ray,
            'tests': [r.to_dict() for r in self.results],
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        report_file = Path("test_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{Colors.OKBLUE}Full report saved to: {report_file}{Colors.ENDC}")
        
        # Final verdict
        print(f"\n{Colors.BOLD}{'='*70}")
        if percentage >= 75:
            print(f"{Colors.OKGREEN}✓ SYSTEM READY FOR PRODUCTION{Colors.ENDC}")
        elif percentage >= 60:
            print(f"{Colors.WARNING}⚠ SYSTEM FUNCTIONAL WITH LIMITATIONS{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}✗ SYSTEM NOT READY - CRITICAL FAILURES{Colors.ENDC}")
        print(f"{'='*70}{Colors.ENDC}\n")


def main():
    """Run test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Labeling System Test Suite')
    parser.add_argument('--no-install', action='store_true', help='Skip automatic dependency installation')
    args = parser.parse_args()
    
    tester = SystemTester(auto_install=not args.no_install)
    
    try:
        tester.run_all_tests()
        
        # Exit code based on score
        percentage = (tester.total_score / tester.max_total_score * 100) if tester.max_total_score > 0 else 0
        
        if percentage >= 75:
            sys.exit(0)  # Success
        elif percentage >= 60:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # Failure
    
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Test interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.FAIL}Fatal error: {e}{Colors.ENDC}")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()