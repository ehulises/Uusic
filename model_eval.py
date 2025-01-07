import json
import os
from typing import List, Tuple
from uusic.common.db import DBInterface
from uusic.common.data_vec import DataVecBuilder
from uusic.common.persona import Persona
from uusic.models.music_classifier import MusicClassifier, MusicClassifierOptions

from pprint import pprint
import numpy as np

from uusic.models.train_embedding import SongEmbeddingModel

PERSONA_FILE = "data/validation/personas.json"
INTERESTING_PERSONA = "data/validation/interesting.json" # what does our model recommend for this persona?
EMBEDDING_PERCENTAGE = 0.9

def evaluate_recommendations(
    recommended: List[str],
    persona: Persona
) -> Tuple[float, float, float]:
    persona_liked = {item.track_id for item in persona.reccomended_tracks}
    persona_disliked = {item.track_id for item in persona.not_reccomended_tracks}
    recommended_set = set(recommended)
    TP = len(recommended_set.intersection(persona_liked))
    FP = len(recommended_set.intersection(persona_disliked))
    FN = len(persona_liked - recommended_set)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def recommend_songs_for_persona(
    persona: Persona,
    builder: DataVecBuilder,
    options: MusicClassifierOptions,
    candidate_tracks: List[str],
    candidate_classifier: MusicClassifier,
    k: int = 20
) -> List[str]:
    user_history_track_ids = [item.track_id for item in persona.listening_history]
    user_history_preferences = [item.rating for item in persona.listening_history]
    user_track_vectors = [None] * len(user_history_track_ids)

    # take each vector, and pass it through the classifier to get the embeddings
    for i, track_id in enumerate(user_history_track_ids):
        raw_vector = builder.build_track_vec(track_id, options.vec_dim)
        if raw_vector is None:
            print(f"Track {track_id} not found in database")
            # remove the preference for this track
            user_history_preferences[i] = 0
            # just use all zeros for the vector
            user_track_vectors[i] = np.zeros(options.embedding_dim, dtype=np.float32)
            continue

        user_track_vectors[i] = candidate_classifier.calculate_embedding(raw_vector)

    # take the weighted average of the user's track vectors based on their preferences
    user_vector = np.zeros(options.embedding_dim, dtype=np.float32)
    for i, vector in enumerate(user_track_vectors):
        # multiply it out by hand because I keep getting a type error
        weighted_vector = [user_history_preferences[i] * v for v in vector]
        user_vector += weighted_vector

    sum_preferences = sum(user_history_preferences)
    if sum_preferences > 0:
        user_vector /= sum_preferences

    # reshape for sklearn
    user_vector = user_vector.reshape(1, -1)

    from sklearn.neighbors import NearestNeighbors
    # depending on the mode, we need to use a different distance metric
    if options.mode == "euclidean":
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
    elif options.mode == "cosine":
        nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")


    nbrs.fit(candidate_classifier.embeddings)


    distances, indices = nbrs.kneighbors(user_vector)
    recommended_indices = indices[0]
    recommended_tracks = [candidate_tracks[i] for i in recommended_indices]
    return recommended_tracks

def main():
    db = DBInterface("data/combined.db")
    builder = DataVecBuilder(db)
    personas = Persona.load_personas_from_json(PERSONA_FILE)

    # Define a list of options to compare
    options_list = [
        MusicClassifierOptions(
            mode="euclidean",
            model_class=SongEmbeddingModel,
            input_dim=100,
            vec_dim=100,
            embedding_dim=32,
            model_path="models/params/song_embedding_euclidean.pth"
        ),
        MusicClassifierOptions(
            mode="cosine",
            model_class=SongEmbeddingModel,
            input_dim=100,
            vec_dim=100,
            embedding_dim=32,
            model_path="models/params/song_embedding_cosine.pth"
        ),
        MusicClassifierOptions(
            mode="euclidean",
            model_class=SongEmbeddingModel,
            input_dim=100,
            vec_dim=100,
            embedding_dim=100,
            model_path=None
        ),
        MusicClassifierOptions(
            mode="cosine",
            model_class=SongEmbeddingModel,
            input_dim=100,
            vec_dim=100,
            embedding_dim=100,
            model_path=None
        )
    ]

    interesting_persona = Persona.load_personas_from_json(INTERESTING_PERSONA)[0]

    # print out the interesting persona
    print("Interesting Persona:")
    for i, track in enumerate(interesting_persona.listening_history):
        db_track = db.get_track_info(track.track_id)
        if db_track is None:
            continue
        print(f"{i}: {db_track['title']} by {db_track['artist_name']}")


    for opts in options_list:
        print(f"Evaluating mode: {opts.mode}")
        
        # Select candidate tracks once per mode
        print("Creating candidate classifier")
        candidate_tracks = db.select_random_tracks_percentage(EMBEDDING_PERCENTAGE)
        candidate_classifier = MusicClassifier(builder=builder, track_ids=candidate_tracks, options=opts)
        print(f"Finished creating candidate classifier with {len(candidate_tracks)} tracks")

        precisions = []
        recalls = []
        f1s = []

        # lets get a list of recommended tracks for our interesting persona
        print("Generating recommendations for interesting persona")
        interesting_recommended_tracks = recommend_songs_for_persona(
            persona=interesting_persona,
            builder=builder,
            options=opts,
            candidate_tracks=candidate_tracks,
            candidate_classifier=candidate_classifier,
            k=20
        )

        for i, track in enumerate(interesting_recommended_tracks):
            db_track = db.get_track_info(track)
            if db_track is None:
                continue
            print(f"{i}: {db_track['title']} by {db_track['artist_name']}")


        for i, persona in enumerate(personas):
            if i % 50 == 0:
                print(f"Generating recommendations for persona: {i}")
            # print(f"User History: {len(persona.listening_history)}")
            # print(f"Recommended: {len(persona.reccomended_tracks)}")
            # print(f"Not Recommended: {len(persona.not_reccomended_tracks)}")

            recommended_tracks = recommend_songs_for_persona(
                persona=persona,
                builder=builder,
                options=opts,
                candidate_tracks=candidate_tracks,
                candidate_classifier=candidate_classifier,
                k=20
            )
            # print(f"Recommended {len(recommended_tracks)} tracks")
            precision, recall, f1 = evaluate_recommendations(recommended_tracks, persona)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            if i % 50 == 0:
                # print out the current average precision, recall, and f1
                avg_precision = np.mean(precisions)
                avg_recall = np.mean(recalls)
                avg_f1 = np.mean(f1s)

                print(f"Mode: {opts.mode} (model {opts.model_path}) - Average Precision: {avg_precision:.2f}")
                print(f"Mode: {opts.mode} (model {opts.model_path}) - Average Recall: {avg_recall:.2f}")
                print(f"Mode: {opts.mode} (model {opts.model_path}) - Average F1: {avg_f1:.2f}\n")

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1s)

        print(f"Mode: {opts.mode} (model {opts.model_path}) - Average Precision: {avg_precision:.2f}")
        print(f"Mode: {opts.mode} (model {opts.model_path}) - Average Recall: {avg_recall:.2f}")
        print(f"Mode: {opts.mode} (model {opts.model_path}) - Average F1: {avg_f1:.2f}\n")

if __name__ == "__main__":
    main()
