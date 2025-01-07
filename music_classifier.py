import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Any, Callable
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

class MusicClassifierOptions:
    """
    Holds configuration for the MusicClassifier.
    """
    def __init__(
        self,
        mode: str, 
        model_class: Callable, 
        input_dim: int, 
        embedding_dim: int, 
        model_path: Optional[str] = None, 
        vec_dim: int = 100
    ):
        """
        Parameters:
        - mode: "euclidean", "cosine", or "raw".
        - model_class: A callable that returns a model instance. Typically a class reference.
        - input_dim: Input dimensionality for the model.
        - embedding_dim: Embedding dimensionality for the model.
        - model_path: Optional path to a trained .pth file.
        - vec_dim: The dimensionality of the raw feature vector extracted from the database.
        """
        self.mode = mode
        self.model_class = model_class
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.vec_dim = vec_dim



class MusicClassifier:
    """
    MusicClassifier uses a MusicClassifierOptions instance to determine how to embed tracks.
    If a model is provided in the options, it loads and uses it. Otherwise, it uses raw features.
    """
    def __init__(
        self, 
        builder: Any,   # Replace `Any` with actual type if available
        track_ids: List[str],
        options: MusicClassifierOptions
    ) -> None:
        """
        Initialize the MusicClassifier using given options.
        """
        self.builder = builder
        self.track_ids = track_ids
        self.options = options

        # Load model if provided
        self.model: Optional[nn.Module] = None
        self.model_loaded = False
        if self.options.model_path and os.path.exists(self.options.model_path):
            self.model = self.options.model_class(self.options.input_dim, self.options.embedding_dim)
            self.model.load_state_dict(torch.load(self.options.model_path, map_location=torch.device('cpu')))
            self.model.eval()
            self.model_loaded = True

        # Precompute embeddings
        self.embeddings = self._compute_embeddings()

    def _track_to_features(self, track_id: str) -> torch.Tensor:
        """
        Extract and prepare features for a single track.
        If a value is None, replace it with 0.0.
        """
        features = self.builder.build_track_vec(track_id, self.options.vec_dim)

        if features is None:
            features = [0.0] * (self.options.input_dim)

        features = [0.0 if v is None else v for v in features]
        # Use only the first input_dim features
        features = features[:self.options.input_dim]
        return torch.tensor(features, dtype=torch.float32)

    def _compute_embeddings(self) -> np.ndarray:
        """
        Compute embeddings for all track IDs.
        If a model is loaded, use it; otherwise, use raw features.
        """
        embeddings = []
        for idx, t_id in enumerate(self.track_ids):
            feat = self._track_to_features(t_id)
            if self.model_loaded and self.model is not None:
                with torch.no_grad():
                    emb = self.model(feat.unsqueeze(0)).squeeze(0).numpy()
            else:
                # Raw features are the embeddings if no model is provided
                emb = feat.numpy()
            embeddings.append(emb)

            if idx % 100 == 0:
                print(f"Computed embeddings for {idx} tracks")

        return np.array(embeddings)
    
    def calculate_embedding(self, data_vector):
        # we need to assert that the data_vector is of our expected length
        assert len(data_vector) == self.options.input_dim

        if self.model is None:
            # if we don't have a model, just return the raw vector
            return data_vector

    
        # pass it through the model
        data_vector = torch.tensor(data_vector, dtype=torch.float32)
        data_vector = data_vector.unsqueeze(0)
        with torch.no_grad():
            emb = self.model(data_vector).squeeze(0).numpy()
        return emb

    def run_knn(
        self, 
        test_size: float = 0.2, 
        n_neighbors: int = 5, 
        random_state: int = 42
    ) -> np.ndarray:
        """
        Run a simple KNN classification on the precomputed embeddings.

        Note: Since we have no true labels, this is a demonstration only.
        """
        embeddings = self.embeddings
        track_id_labels = self.track_ids

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, track_id_labels, test_size=test_size, random_state=random_state
        )

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test[:5])
        print("KNN Predictions on the first 5 test samples:", predictions)
        return predictions

    def run_kmeans(
        self, 
        n_clusters: int = 10, 
        random_state: int = 42
    ) -> Dict[str, int]:
        """
        Run K-Means clustering on the precomputed embeddings.
        """
        embeddings = self.embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(embeddings)

        track_id_to_cluster = {
            track_id: cluster for track_id, cluster in zip(self.track_ids, kmeans.labels_)
        }
        print("K-Means Clusters (First 10 Tracks):", list(track_id_to_cluster.items())[:10])
        return track_id_to_cluster
