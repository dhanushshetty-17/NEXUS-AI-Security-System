import threading
from typing import Any

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    _REID_AVAILABLE = True
except ImportError:
    np = None
    cosine_similarity = None
    _REID_AVAILABLE = False

class GlobalReIDManager:
    """
    Maintains a global gallery of identities across multiple cameras.
    Assigns a persistent Global-ID based on visual appearance features (embeddings).
    """
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        # Maps global_id (str) to a numpy array representing the moving average of their embedding
        self.gallery: dict[str, Any] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def assign_global_id(self, features: Any) -> str:
        """
        Takes a feature vector (e.g. from DeepSORT) and returns a matching global ID.
        If no match is found above the threshold, creates a new one.
        """
        if not _REID_AVAILABLE:
            with self._lock:
                return self._create_new_id(features)

        # DeepSORT sometimes returns a list of features, we'll take the mean if it's 2D
        if hasattr(features, "shape") and len(features.shape) > 1:
            features = np.mean(features, axis=0)
            
        features = features.reshape(1, -1)
        
        with self._lock:
            if not self.gallery:
                return self._create_new_id(features)

            # Compare against all known global identities
            global_ids = list(self.gallery.keys())
            gallery_matrix = np.vstack(list(self.gallery.values()))
            
            similarities = cosine_similarity(features, gallery_matrix)[0]
            
            best_idx = int(np.argmax(similarities))
            best_score = similarities[best_idx]

            if best_score >= self.similarity_threshold:
                matched_id = global_ids[best_idx]
                # Update moving average slightly to adapt to appearance changes (e.g. lighting)
                alpha = 0.1
                self.gallery[matched_id] = (1 - alpha) * self.gallery[matched_id] + alpha * features[0]
                return matched_id
            else:
                return self._create_new_id(features)

    def _create_new_id(self, features: Any) -> str:
        new_id = f"G-{self._next_id:03d}"
        if _REID_AVAILABLE and hasattr(features, "__getitem__"):
            self.gallery[new_id] = features[0] if len(features.shape) > 1 else features
        self._next_id += 1
        return new_id
