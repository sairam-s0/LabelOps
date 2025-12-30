# src/core/model_manager.py
import shutil
import os
import tempfile
from pathlib import Path
from datetime import datetime

# Lazy import YOLO to avoid dependency issues unless needed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ModelManager:
    def __init__(self, models_dir="models", base_model_name="yolov8m.pt"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.base_model_name = base_model_name
        self.active_symlink = self.models_dir / "active_model.pt"
        self._cached_model_path = None  # Cache for performance
        
        # Ensure versions dir exists
        (self.models_dir / "versions").mkdir(exist_ok=True)

    def _is_temp_directory(self, path: Path) -> bool:
        """Check if we're in a temporary directory (skip deep search)."""
        temp_prefixes = [tempfile.gettempdir()]
        str_path = str(path.resolve())
        return any(str_path.startswith(prefix) for prefix in temp_prefixes)

    def _find_project_root(self, start_path: Path = None) -> Path:
        """
        Walk up from start_path to find project root.
        Root is identified by: .git/, src/, 'dynamic' in name, or requirements.txt.
        """
        if start_path is None:
            start_path = Path.cwd()
        
        current = start_path.resolve()
        fs_root = Path(current.root)
        
        while current != fs_root:
            # Check for project markers
            if (current / ".git").exists():
                return current
            if (current / "src").exists():
                return current
            if (current / "requirements.txt").exists():
                return current
            if current.name == "dynamic":
                return current
            current = current.parent
        
        return fs_root  # Fallback to filesystem root

    def find_model_file(self, model_name: str) -> str:
        """
        Search for model file with smart project-aware logic.
        Returns full path as string, or None if not found.
        """
        if self._cached_model_path:
            return self._cached_model_path

        # Quick search paths (priority order)
        search_paths = [
            self.models_dir / model_name,          # 1. models/ directory
            Path.cwd() / model_name,               # 2. current directory
            Path.cwd().parent / model_name,        # 3. parent directory
        ]
        
        for path in search_paths:
            if path.exists():
                self._cached_model_path = str(path.resolve())
                return self._cached_model_path

        # Skip deep search in temp directories
        if self._is_temp_directory(Path.cwd()):
            return None

        # Deep recursive search from project root
        try:
            project_root = self._find_project_root()
            # Search in project root and subdirectories
            for found in project_root.rglob(f"**/{model_name}"):
                if found.is_file():
                    self._cached_model_path = str(found.resolve())
                    return self._cached_model_path
        except Exception as e:
            print(f"[ModelManager] Deep search failed: {e}")

        return None

    def download_model(self, model_name: str) -> str:
        """
        Download model using Ultralytics YOLO, then move to models/ directory.
        Returns path to downloaded model.
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics YOLO not available for auto-download")
        
        print(f"[ModelManager] Downloading {model_name}...")
        try:
            # This downloads to current directory
            model = YOLO(model_name)
            downloaded_path = Path.cwd() / model_name
            
            if not downloaded_path.exists():
                raise FileNotFoundError(f"Downloaded model not found at {downloaded_path}")
            
            # Move to models/ directory for organization
            target_path = self.models_dir / model_name
            shutil.move(str(downloaded_path), str(target_path))
            
            print(f"[ModelManager] Downloaded {model_name} to {target_path}")
            return str(target_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download {model_name}: {e}")

    def get_active_model(self) -> str:
        """
        Returns path to the symlink acting as the active model.
        Ensures base model exists (searches or downloads if needed).
        """
        if not self.active_symlink.exists():
            # Try to find existing model
            found_path = self.find_model_file(self.base_model_name)
            
            if found_path is None:
                # Auto-download if not found
                if YOLO_AVAILABLE:
                    found_path = self.download_model(self.base_model_name)
                else:
                    raise FileNotFoundError(
                        f"Base model '{self.base_model_name}' not found. "
                        f"Searched in models/, current dir, and project directories. "
                        f"Please place it in the 'models/' directory or install ultralytics for auto-download."
                    )
            
            # Ensure base_model points to found/downloaded path
            self.base_model = Path(found_path)
            
            # Create initial symlink
            try:
                os.symlink(self.base_model.resolve(), self.active_symlink)
            except OSError:
                # Windows fallback if no admin rights for symlinks
                shutil.copy2(self.base_model, self.active_symlink)
                
        return str(self.active_symlink)

    def resolve_active_path(self) -> str:
        """
        Returns the actual .pt file path that the active symlink points to.
        - If active_symlink is a valid symlink: returns its target (resolved).
        - If it's a regular file (e.g., Windows fallback): returns its own path.
        - Ensures symlink exists by calling get_active_model() first.
        """
        # Ensure active symlink/file exists
        self.get_active_model()  # side effect: creates it if missing

        active_path = self.active_symlink

        # Check if it's a symlink and points to something valid
        if active_path.is_symlink():
            try:
                resolved = active_path.resolve()
                if resolved.exists():
                    return str(resolved)
                else:
                    # Broken symlink — fall back to treating as regular file
                    return str(active_path)
            except Exception:
                # Fallback on any resolution error
                return str(active_path)
        else:
            # Not a symlink (e.g., copied file on Windows)
            return str(active_path)

    def promote_shadow(self, shadow_path: str) -> dict:
        """
        Promote a shadow model to active status atomically.
        """
        shadow_file = Path(shadow_path)
        if not shadow_file.exists():
            return {'success': False, 'error': 'Shadow file not found'}

        # 1. Version the new model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"v_{timestamp}.pt"
        target_path = self.models_dir / "versions" / version_name
        
        shutil.copy2(shadow_file, target_path)

        # 2. Update Symlink Atomically
        try:
            # Create temp link then rename (atomic on POSIX, usually safe on modern Windows)
            tmp_link = self.models_dir / "tmp_active.pt"
            if tmp_link.exists():
                os.remove(tmp_link)
            
            os.symlink(target_path.resolve(), tmp_link)
            os.replace(tmp_link, self.active_symlink)
            
        except OSError:
            # Fallback for systems without symlink permission
            shutil.copy2(target_path, self.active_symlink)

        return {
            'success': True,
            'version': version_name,
            'path': str(target_path),
            'timestamp': timestamp
        }

    def rollback(self, specific_version: str = None) -> dict:
        """
        Revert to previous version.
        """
        versions_dir = self.models_dir / "versions"
        if not versions_dir.exists():
            return {'success': False, 'error': 'No versions directory'}

        versions = sorted(versions_dir.glob("*.pt"), key=os.path.getmtime)
        
        if not versions:
            return {'success': False, 'error': 'No versions to rollback to'}

        target = None
        if specific_version:
            # Find specific version by name
            for v in versions:
                if v.name == specific_version:
                    target = v
                    break
        else:
            # Rollback to 2nd most recent (current is most recent)
            if len(versions) >= 2:
                target = versions[-2]
            else:
                target = versions[0]  # Fallback to only one existing

        if target is None:
            return {'success': False, 'error': 'Target version not found'}

        # Update symlink
        try:
            tmp_link = self.models_dir / "tmp_active.pt"
            if tmp_link.exists():
                os.remove(tmp_link)
            os.symlink(target.resolve(), tmp_link)
            os.replace(tmp_link, self.active_symlink)
        except OSError:
            shutil.copy2(target, self.active_symlink)

        return {'success': True, 'target': target.name}

    def list_versions(self) -> list:
        versions_dir = self.models_dir / "versions"
        versions = versions_dir.glob("*.pt")
        return [
            {
                "name": v.name,
                "size_mb": round(v.stat().st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(v.stat().st_ctime).isoformat()
            }
            for v in sorted(versions, key=os.path.getmtime, reverse=True)
        ]