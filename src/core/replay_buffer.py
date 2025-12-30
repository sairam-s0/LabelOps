# src/core/replay_buffer.py
import random
from collections import defaultdict
from datetime import datetime, timedelta

class ReplayBuffer:
    def __init__(self, max_size=200, max_age_days=30):
        self.buffer = []  # List of sample dicts
        self.max_size = max_size
        self.max_age_days = max_age_days
        self.class_counts = defaultdict(int)
    def __len__(self):
        return len(self.buffer)

    def add(self, samples):
        """
        Add samples to replay buffer with deduplication.
        
        Args:
            samples: a single dict OR list of dicts with structure:
                {
                    'image_path': str,
                    'entropy': float,
                    'detections': list,
                    'timestamp': str (ISO format, optional),
                    'width': int,
                    'height': int
                }
        """
        # 🔧 STEP 1: Normalize input to list of dicts
        if isinstance(samples, dict):
            samples = [samples]
        elif not isinstance(samples, list):
            print("[ReplayBuffer] WARNING: 'samples' must be a dict or list of dicts. Skipping.")
            return

        added_count = 0
        
        for s in samples:
            # 🔧 STEP 2: Validate that s is a dict
            if not isinstance(s, dict):
                print(f"[ReplayBuffer] WARNING: Non-dict sample encountered (type: {type(s)}). Skipping.")
                continue

            # Ensure required field exists
            if 'image_path' not in s:
                print("[ReplayBuffer] WARNING: Sample missing 'image_path'. Skipping.")
                continue

            # Deduplication check
            if any(x.get('image_path') == s['image_path'] for x in self.buffer):
                continue

            # 🔧 STEP 3: Add timestamp if missing
            if 'timestamp' not in s:
                s['timestamp'] = datetime.now().isoformat()

            self.buffer.append(s)
            added_count += 1

            # Update class counts
            for det in s.get('detections', []):
                cls_name = det.get('class_name') or det.get('class')
                if cls_name:
                    self.class_counts[cls_name] += 1

        # Cleanup if over capacity
        if len(self.buffer) > self.max_size:
            self._prune()
        
        if added_count > 0:
            print(f"[ReplayBuffer] Added {added_count} samples, total: {len(self.buffer)}")

    def _prune(self):
        """
        Remove samples to maintain max_size.
        Strategy: Remove lowest entropy samples first, but respect temporal decay.
        """
        now = datetime.now()
        # Add age and priority metadata
        for s in self.buffer:
            try:
                ts = datetime.fromisoformat(s['timestamp'])
                age_days = (now - ts).days
                s['_age_days'] = age_days
            except (ValueError, KeyError):
                s['_age_days'] = 0

        # Apply temporal decay to entropy
        for s in self.buffer:
            age = s['_age_days']
            decay = max(0.5, 1.0 - (age / (self.max_age_days * 2)))
            s['_priority'] = s.get('entropy', 0.0) * decay

        # Sort by priority (keep highest)
        self.buffer.sort(key=lambda x: x.get('_priority', 0.0))

        # Remove lowest priority samples
        remove_count = len(self.buffer) - self.max_size
        if remove_count > 0:
            removed = self.buffer[:remove_count]
            self.buffer = self.buffer[remove_count:]

            # Update class counts
            for s in removed:
                for det in s.get('detections', []):
                    cls_name = det.get('class_name') or det.get('class')
                    if cls_name:
                        self.class_counts[cls_name] = max(0, self.class_counts[cls_name] - 1)

            print(f"[ReplayBuffer] Pruned {remove_count} samples")

    def sample(self, count=10, strategy='entropy') -> list:
        """
        Sample from replay buffer using various strategies.
        """
        if not self.buffer:
            return []
        
        count = min(count, len(self.buffer))

        if strategy == 'random':
            return random.sample(self.buffer, count)
        
        elif strategy == 'entropy':
            sorted_buf = sorted(
                self.buffer, 
                key=lambda x: x.get('entropy', 0.0), 
                reverse=True
            )
            return sorted_buf[:count]
        
        elif strategy == 'recent':
            sorted_buf = sorted(
                self.buffer,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )
            return sorted_buf[:count]
            
        elif strategy == 'balanced':
            return self._balanced_sample(count)
        
        return self.sample(count, strategy='entropy')

    def _balanced_sample(self, count: int) -> list:
        """
        Class-balanced sampling to prevent catastrophic forgetting.
        """
        if not self.buffer:
            return []
        
        class_samples = defaultdict(list)
        for s in self.buffer:
            for det in s.get('detections', []):
                cls_name = det.get('class_name') or det.get('class')
                if cls_name:
                    class_samples[cls_name].append(s)
        
        if not class_samples:
            return random.sample(self.buffer, min(count, len(self.buffer)))
        
        selected = []
        selected_paths = set()
        total_class_samples = sum(len(samples) for samples in class_samples.values())

        for cls_name, samples in class_samples.items():
            class_proportion = len(samples) / total_class_samples
            class_count = max(1, int(count * class_proportion))
            
            samples_sorted = sorted(
                samples, 
                key=lambda x: x.get('entropy', 0.0), 
                reverse=True
            )
            
            for s in samples_sorted[:class_count]:
                if s['image_path'] not in selected_paths:
                    selected.append(s)
                    selected_paths.add(s['image_path'])
                if len(selected) >= count:
                    break
            if len(selected) >= count:
                break
        
        # Fill remaining slots if needed
        if len(selected) < count:
            remaining = [s for s in self.buffer if s['image_path'] not in selected_paths]
            needed = count - len(selected)
            if remaining:
                selected.extend(random.sample(remaining, min(needed, len(remaining))))
        
        return selected[:count]

    def get_class_distribution(self) -> dict:
        return dict(self.class_counts)

    def get_stats(self) -> dict:
        if not self.buffer:
            return {
                'size': 0,
                'capacity': self.max_size,
                'utilization': 0.0,
                'class_distribution': {}
            }
        
        entropies = [s.get('entropy', 0.0) for s in self.buffer]
        now = datetime.now()
        ages = []
        for s in self.buffer:
            try:
                ts = datetime.fromisoformat(s.get('timestamp', ''))
                ages.append((now - ts).days)
            except (ValueError, TypeError):
                ages.append(0)
        
        return {
            'size': len(self.buffer),
            'capacity': self.max_size,
            'utilization': len(self.buffer) / self.max_size,
            'class_distribution': dict(self.class_counts),
            'avg_entropy': sum(entropies) / len(entropies),
            'max_entropy': max(entropies),
            'min_entropy': min(entropies),
            'avg_age_days': sum(ages) / len(ages) if ages else 0,
            'max_age_days': max(ages) if ages else 0
        }

    def clear(self):
        self.buffer.clear()
        self.class_counts.clear()
        print("[ReplayBuffer] Cleared all samples")

    def remove_old_samples(self, max_age_days: int = None):
        if max_age_days is None:
            max_age_days = self.max_age_days
        
        cutoff = datetime.now() - timedelta(days=max_age_days)
        initial_size = len(self.buffer)
        
        filtered_buffer = []
        for s in self.buffer:
            try:
                ts_str = s.get('timestamp')
                if ts_str:
                    ts = datetime.fromisoformat(ts_str)
                else:
                    ts = datetime.now()  # treat missing as now
                if ts > cutoff:
                    filtered_buffer.append(s)
            except (ValueError, TypeError):
                # Keep samples with invalid timestamps (conservative)
                filtered_buffer.append(s)
        
        # Rebuild class counts
        self.buffer = filtered_buffer
        self.class_counts.clear()
        for s in self.buffer:
            for det in s.get('detections', []):
                cls_name = det.get('class_name') or det.get('class')
                if cls_name:
                    self.class_counts[cls_name] += 1
        
        removed = initial_size - len(self.buffer)
        if removed > 0:
            print(f"[ReplayBuffer] Removed {removed} samples older than {max_age_days} days")