import numpy as np
from typing import List, Dict
from random import choice
from uusic.common.db import DBInterface
import json


class ListeningHistoryItem:
    def __init__(self, track_id: str, rating: float, name: str = None):
        assert track_id is not None, "track_id cannot be None"
        assert rating is not None, "rating cannot be None"
        assert isinstance(track_id, str), "track_id must be a string"
        assert isinstance(rating, float), "rating must be a float"
        assert 0.0 <= rating <= 1.0, "rating must be between 0.0 and 1.0"
        assert name is None or isinstance(name, str), "name must be a string or None"

        self.track_id = track_id
        self.track_name = name
        self.rating = rating

    def __repr__(self):
        return (
            f"ListeningHistoryItem({self.track_id}, {self.rating}, {self.track_name})"
        )

    def __str__(self):
        return f"{self.track_id} ({self.rating}, {self.track_name})"

    def __eq__(self, other):
        return self.track_id == other.track_id and self.rating == other.rating

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.track_id, self.rating))

    def __dict__(self):
        return {
            "track_id": self.track_id,
            "rating": self.rating,
            "track_name": self.track_name,
        }


class ReccommendedItem:
    def __init__(self, track_id: str, name: str = None):
        assert track_id is not None, "track_id cannot be None"
        assert isinstance(track_id, str), "track_id must be a string"
        assert name is None or isinstance(name, str), "name must be a string or None"

        self.track_id = track_id
        self.track_name = name

    def __repr__(self):
        return f"ReccommendedItem({self.track_id}, {self.track_name})"

    def __str__(self):
        return f"{self.track_id} ({self.track_name})"

    def __eq__(self, other):
        return self.track_id == other.track_id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.track_id)

    def __dict__(self):
        return {"track_id": self.track_id, "track_name": self.track_name}


class Persona:
    def __init__(self):
        self.listening_history: List[ListeningHistoryItem] = []
        self.reccomended_tracks: List[ReccommendedItem] = []
        self.not_reccomended_tracks: List[ReccommendedItem] = []

    @staticmethod
    def build_persona_algorithmic(
        num_user_history: int,
        num_reccomended: int,
        num_not_reccomended: int,
        db: DBInterface,
    ) -> "Persona":

        persona = Persona()

        similarity_cache: Dict[str, Dict[str, float]] = {}

        def get_similarity(track_a: str, track_b: str) -> float:
            # Check if similarity is cached
            if track_a in similarity_cache and track_b in similarity_cache[track_a]:
                return similarity_cache[track_a][track_b]

            similarity = db.get_similarity(track_a, track_b)
            if similarity is None:
                similarity = 0.0

            # Cache similarity
            if track_a not in similarity_cache:
                similarity_cache[track_a] = {}
            similarity_cache[track_a][track_b] = similarity

            return similarity

        def weighted_avg_similarity(candidate_id: str, history: List[ListeningHistoryItem]) -> float:
            if not history:
                return 0.0
            # Extract ratings and compute similarities
            ratings = np.array([h.rating for h in history])
            track_ids = [h.track_id for h in history]
            
            # Compute similarities for candidate to each track in history
            sims = np.array([get_similarity(candidate_id, h_id) for h_id in track_ids])
            
            # Weighted average
            if ratings.sum() == 0:
                return 0.0
            return np.average(sims, weights=ratings)

        # print("Starting to build a single persona...")

        # Step 1: Pick a random track as seed
        seed_track = db.select_random_tracks_percentage(1)[0]
        seed_rating = float(np.random.uniform(0.5, 1.0))
        persona.add_listening_history(
            seed_track, seed_rating, db.get_track_title(seed_track)
        )
        # print(f"Seed track chosen: {seed_track} with rating {seed_rating:.2f}")

        # Step 2: From the seed, pick top 10 similar tracks
        similar_to_seed = db.get_similar_tracks(seed_track)
        top_10_seed = similar_to_seed[:10] if len(similar_to_seed) > 10 else similar_to_seed
        if top_10_seed:
            chosen = choice(top_10_seed)
            chosen_track_id, _ = chosen
            chosen_rating = float(np.random.uniform(0.5, 1.0))
            persona.add_listening_history(
                chosen_track_id, chosen_rating, db.get_track_title(chosen_track_id)
            )
            # print(f"Added a similar track to seed: {chosen_track_id} with rating {chosen_rating:.2f}")

        # Step 3: Expand the listening history until we have num_user_history items
        while len(persona.listening_history) < num_user_history:
            current_count = len(persona.listening_history)
            # print(f"Current listening history size: {current_count}. Target: {num_user_history}")

            track_id_list = [h.track_id for h in persona.listening_history]
            candidates = db.get_similar_tracks_from_list(track_id_list, 0.1)

            # remove tracks already in history
            candidates = [c for c in candidates if c not in track_id_list]
 
            if not candidates:
                # do a tad bit of exploration, and add a random track
                random_track = db.select_random_tracks_percentage(1)[0]
                random_rating = float(np.random.uniform(0.5, 1.0))
                persona.add_listening_history(
                    random_track, random_rating, db.get_track_title(random_track)
                )

                print(f"Added random track {random_track} with rating {random_rating:.2f}")
                continue
            

            scores = []
            for idx, c in enumerate(candidates):
                wav_sim = weighted_avg_similarity(c, persona.listening_history)
                scores.append((c, wav_sim))

                # if idx % 10 == 0:
                #     print(f"Processed {idx}/{len(candidates)} candidates...")

            dtype = [('track_id', 'U100'), ('score', 'f8')]
            scores_arr = np.array(scores, dtype=dtype)
            scores_arr = np.sort(scores_arr, order='score')[::-1]

            # print(f"Top 10 candidates: {scores_arr[:10]}")

            top_10 = scores_arr[:10] if len(scores_arr) > 10 else scores_arr
            if len(top_10) == 0:
                print("No top candidates found.")
                break

            # choose the top 10, weighted by their similarity
            weights = [t['score'] for t in top_10]
            chosen_candidate = np.random.choice(top_10, p=weights/np.sum(weights))

            chosen_id, chosen_wav_sim = chosen_candidate['track_id'], chosen_candidate['score']

            # find the max similarity and use that as a base for the rating
            max_sim = max([get_similarity(chosen_id, h.track_id) for h in persona.listening_history])

            # now randomly distribute the rating between the max similarity and the weighted average similarity
            chosen_rating = float(np.random.uniform(chosen_wav_sim, max_sim))
            persona.add_listening_history(
                chosen_id, chosen_rating, db.get_track_title(chosen_id)
            )
            # print(f"Added track {chosen_id} from top candidates with rating {chosen_rating:.2f}")

        # Step 4: Once we have our desired listening history, find recommended and not recommended tracks
        # print("Finding recommended and not recommended tracks...")
        history_ids = [h.track_id for h in persona.listening_history]
        candidates = db.get_similar_tracks_from_list(history_ids, 0.0)

        # remove tracks already in history
        candidates = [c for c in candidates if c not in history_ids]
        scores = []

        for idx, c in enumerate(candidates):
            wav_sim = weighted_avg_similarity(c, persona.listening_history)
            scores.append((c, wav_sim))

            # if idx % 10 == 0:
            #     print(f"Processed {idx}/{len(candidates)} candidates...")

        dtype = [('track_id', 'U100'), ('score', 'f8')]
        scores_arr = np.array(scores, dtype=dtype)
        scores_arr = np.sort(scores_arr, order='score')[::-1]

        recommended = scores_arr[:num_reccomended]
        not_recommended = scores_arr[-num_not_reccomended:] if num_not_reccomended > 0 else []

        # only allow tracks in one of the two lists
        not_recommended = [nr for nr in not_recommended if nr not in recommended]

        for r_item in recommended:
            persona.add_reccomended_track(r_item['track_id'], db.get_track_title(r_item['track_id']))
        for nr_item in not_recommended:
            persona.add_not_reccomended_track(nr_item['track_id'], db.get_track_title(nr_item['track_id']))

        print(f"Built persona with {len(persona.listening_history)} listening history items, "
              f"{len(persona.reccomended_tracks)} recommended tracks, "
              f"and {len(persona.not_reccomended_tracks)} not recommended tracks.")

        return persona

    @staticmethod
    def build_persona_set_algorithmic(
        num_personas: int,
        num_user_history: int,
        num_reccomended: int,
        num_not_reccomended: int,
        db: DBInterface,
    ) -> List["Persona"]:
        print(f"Building {num_personas} personas...")
        personas = []
        for i in range(num_personas):
            print(f"Building persona {i+1}/{num_personas}...")
            persona = Persona.build_persona_algorithmic(
                num_user_history, num_reccomended, num_not_reccomended, db
            )
            personas.append(persona)
        print(f"All {num_personas} personas built successfully.")
        return personas

    @staticmethod
    def load_personas_from_json(json_file: str) -> List["Persona"]:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            personas = [Persona.load_from_dict(p) for p in data]
        return personas
    
    @staticmethod
    def load_from_dict(data: Dict) -> "Persona":
        p = Persona()
        for lh in data["listening_history"]:
            p.add_listening_history(lh["track_id"], lh["rating"], lh["track_name"])
        for rt in data["reccomended_tracks"]:
            p.add_reccomended_track(rt["track_id"], rt["track_name"])
        for nrt in data["not_reccomended_tracks"]:
            p.add_not_reccomended_track(nrt["track_id"], nrt["track_name"])

        return p

    def add_listening_history(self, track_id: str, rating: float, name: str = None):
        self.listening_history.append(ListeningHistoryItem(track_id, rating, name))

    def add_reccomended_track(self, track_id: str, name: str = None):
        self.reccomended_tracks.append(ReccommendedItem(track_id, name))

    def add_not_reccomended_track(self, track_id: str, name: str = None):
        self.not_reccomended_tracks.append(ReccommendedItem(track_id, name))

    def __repr__(self):
        return f"Persona({self.listening_history}, {self.reccomended_tracks}, {self.not_reccomended_tracks})"

    def __str__(self):
        return f"Persona with {len(self.listening_history)} listening history items, {len(self.reccomended_tracks)} reccomended tracks, and {len(self.not_reccomended_tracks)} not reccomended tracks"

    def __eq__(self, other):
        return (
            self.listening_history == other.listening_history
            and self.reccomended_tracks == other.reccomended_tracks
            and self.not_reccomended_tracks == other.not_reccomended_tracks
        )

    def __dict__(self):
        return {
            "listening_history": [item.__dict__() for item in self.listening_history],
            "reccomended_tracks": [item.__dict__() for item in self.reccomended_tracks],
            "not_reccomended_tracks": [
                item.__dict__() for item in self.not_reccomended_tracks
            ],
        }
