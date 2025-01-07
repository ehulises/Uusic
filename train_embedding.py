import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from uusic.common.db import DBInterface
from uusic.common.data_vec import DataVecBuilder

#############################################
# Configuration and Options
#############################################
DB_PATH = "data/combined.db"
VEC_DIM = 100  # Dimension of the raw feature vector extracted per track
INPUT_DIM = 100  # Input dimensionality for the model
EMBEDDING_DIM = 32  # Embedding dimension
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
N_NEIGHBORS = 5
N_CLUSTERS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
DATA_PERCENT = 0.8

MODEL_PATH_EUCLIDEAN = "models/params/song_embedding_euclidean.pth"
MODEL_PATH_COSINE = "models/params/song_embedding_cosine.pth"


#############################################
# Data Preparation Functions
#############################################
def track_to_features(builder, track_id):
    """
    Convert a track ID into a feature vector. Replace None values with 0.0.
    Returns a torch.FloatTensor of shape [INPUT_DIM].
    """
    features = builder.build_track_vec(track_id, VEC_DIM)
    features = [0.0 if v is None else v for v in features]
    features = features[:INPUT_DIM]
    return torch.tensor(features, dtype=torch.float32)


def get_similarity_pairs(db, track_ids):
    """
    Fetch (track_id_a, track_id_b, similarity) pairs for the given track_ids.
    Only include pairs where both tracks are in track_ids.
    """
    pairs = []
    for t_id in track_ids:
        sim_tracks = db.get_similar_tracks(t_id)
        for other_id, sim_val in sim_tracks:
            if other_id in track_ids:
                pairs.append((t_id, other_id, sim_val))
    return pairs


#############################################
# Dataset Definition
#############################################
class TrackPairDataset(Dataset):
    """
    Dataset for track pairs and their similarity scores.
    Each item is (feature_a, feature_b, similarity).
    """

    def __init__(self, pairs, builder):
        self.pairs = pairs
        self.builder = builder

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        t_id_a, t_id_b, sim = self.pairs[idx]
        feat_a = track_to_features(self.builder, t_id_a)
        feat_b = track_to_features(self.builder, t_id_b)
        if feat_a is None or feat_b is None:
            raise ValueError("Missing features for track.")
        return feat_a, feat_b, torch.tensor(sim, dtype=torch.float32)


#############################################
# Model Definition
#############################################
class SongEmbeddingModel(nn.Module):
    """
    Model to generate embeddings for tracks from input features.
    """

    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.fc(x)


#############################################
# Training and Evaluation Functions
#############################################
def train_model(model, criterion, optimizer, dataloader, epochs, metric):
    """
    Train the model using the specified metric ("euclidean" or "cosine").

    If metric == "euclidean":
        - Compute squared Euclidean distance between embeddings.
        - The target distance is (1 - similarity).

    If metric == "cosine":
        - Compute cosine similarity between embeddings.
        - The target is the similarity itself.
    """
    model.train()
    last_val_loss = 1000.0
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for feat_a, feat_b, sim in dataloader:
            optimizer.zero_grad()

            emb_a = model(feat_a)
            emb_b = model(feat_b)

            if metric == "euclidean":
                dist_sq = torch.sum((emb_a - emb_b) ** 2, dim=1)
                target = 1.0 - sim
                loss = criterion(dist_sq, target)
            elif metric == "cosine":
                cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1)
                target = sim
                loss = criterion(cos_sim, target)
            else:
                raise ValueError("Unknown metric specified.")

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feat_a.size(0)
            count += feat_a.size(0)

        avg_loss = total_loss / count
        print(f"[{metric.capitalize()}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        last_val_loss = avg_loss


        # now validate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_count = 0
            for feat_a, feat_b, sim in dataloader:
                emb_a = model(feat_a)
                emb_b = model(feat_b)
                if metric == "euclidean":
                    dist_sq = torch.sum((emb_a - emb_b) ** 2, dim=1)
                    target = 1.0 - sim
                    loss = criterion(dist_sq, target)
                elif metric == "cosine":
                    cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1)
                    target = sim
                    loss = criterion(cos_sim, target)
                val_loss += loss.item() * feat_a.size(0)
                val_count += feat_a.size(0)
            avg_val_loss = val_loss / val_count
            print(f"[{metric.capitalize()}] Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < 0.1:
            print("Early stopping at epoch", epoch+1)
            break

        if avg_val_loss > last_val_loss:
            print("Early stopping at epoch", epoch+1)

        if last_val_loss - avg_val_loss < 0.01:
            print("Early stopping at epoch", epoch+1)
            break

        # reset the model back to training mode
        model.train()
        


def generate_embeddings_with_labels(model, builder, track_ids):
    """
    Generate embeddings for given track IDs using the trained model.
    Returns:
        embeddings: List of numpy arrays (one per track_id).
        track_id_labels: Corresponding list of track IDs.
    """
    model.eval()
    embeddings = []
    track_id_labels = []
    with torch.no_grad():
        for track_id in track_ids:
            features = track_to_features(builder, track_id)
            if features is not None:
                embedding = model(features.unsqueeze(0)).squeeze(0)
                embeddings.append(embedding.numpy())
                track_id_labels.append(track_id)
    return embeddings, track_id_labels


def evaluate_with_knn(embeddings, labels):
    """
    Run a K-Nearest Neighbors classification on the embeddings for a quick check.
    Prints predictions for a small sample.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test[:5])
    print(f"KNN Predictions: {predictions}")


def evaluate_with_kmeans(embeddings, labels):
    """
    Run K-Means clustering on the embeddings for a quick check.
    Prints the clusters for the first 10 tracks.
    """
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    kmeans.fit(embeddings)
    track_id_to_cluster = {
        track_id: cluster for track_id, cluster in zip(labels, kmeans.labels_)
    }
    print("K-Means Clusters (First 10 Tracks):")
    print(list(track_id_to_cluster.items())[:10])


#############################################
# Model Persistence Functions
#############################################
def save_model(model, path):
    """
    Save the model state dictionary to a file.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model_class, input_dim, embedding_dim, path):
    """
    Load the model state dictionary from a file.
    Returns an instance of model_class with loaded parameters.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model file found at {path}")
    model = model_class(input_dim, embedding_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model


#############################################
# Main Functionality
#############################################
def prepare_data():
    """
    Prepare the dataset and dataloader for training.
    """
    db = DBInterface(db_path=DB_PATH)
    builder = DataVecBuilder(db)
    data_split = builder.get_training_split(DATA_PERCENT)
    train_track_ids = data_split[0]

    pairs = get_similarity_pairs(db, train_track_ids)
    dataset = TrackPairDataset(pairs, builder)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return builder, train_track_ids, dataloader


def train_and_save_for_metric(metric, builder, train_track_ids, dataloader, model_path):
    """
    Train a model for the given metric (euclidean or cosine) and save it.
    """
    print(f"Training with {metric.capitalize()} Metric...")
    model = SongEmbeddingModel(INPUT_DIM, EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_model(model, criterion, optimizer, dataloader, EPOCHS, metric=metric)
    save_model(model, model_path)

    embeddings, labels = generate_embeddings_with_labels(
        model, builder, train_track_ids
    )
    print(f"Evaluating {metric.capitalize()} model with KNN...")
    evaluate_with_knn(embeddings, labels)

    print(f"Evaluating {metric.capitalize()} model with K-Means...")
    evaluate_with_kmeans(embeddings, labels)


def load_and_evaluate_model(model_path, builder, track_ids):
    """
    Load a saved model and evaluate it using KNN and K-Means.
    """
    model = load_model(SongEmbeddingModel, INPUT_DIM, EMBEDDING_DIM, model_path)
    embeddings, labels = generate_embeddings_with_labels(model, builder, track_ids)

    print("Evaluating loaded model with KNN...")
    evaluate_with_knn(embeddings, labels)

    print("Evaluating loaded model with K-Means...")
    evaluate_with_kmeans(embeddings, labels)


def main():
    # Prepare data
    builder, train_track_ids, dataloader = prepare_data()

    # Train and save Euclidean model
    train_and_save_for_metric(
        "euclidean", builder, train_track_ids, dataloader, MODEL_PATH_EUCLIDEAN
    )

    # Train and save Cosine model
    train_and_save_for_metric(
        "cosine", builder, train_track_ids, dataloader, MODEL_PATH_COSINE
    )

    # Example of loading a model and evaluating
    # (You can comment this out if you don't want to run it every time)
    load_and_evaluate_model(MODEL_PATH_EUCLIDEAN, builder, train_track_ids)
    load_and_evaluate_model(MODEL_PATH_COSINE, builder, train_track_ids)


if __name__ == "__main__":
    main()
