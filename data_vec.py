# data_vec.py

from uusic.common.db import DBInterface
from typing import List


class DataVecBuilder:
    def __init__(self, db: DBInterface):
        self.db: DBInterface = db
        self.tag_cache = []

    def _build_tag_cache(self, count):
        self.tag_cache = self.db.get_common_tags(count)

    def build_track_vec(self, track_id: str, tag_count: int) -> List[float]:
        if len(self.tag_cache) < tag_count:
            self._build_tag_cache(tag_count)

        info = self.db.get_track_info(track_id)
        if info is None:
            return None
        features = [
            info["danceability"],
            info["duration"],
            info["energy"],
            info["key"],
            info["loudness"],
            info["mode"],
            info["tempo"],
            info["time_signature"],
            info["year"],
        ]

        track_tags = self.db.get_track_tags(track_id)
        tag_vec = [0] * tag_count

        for idx, tag in enumerate(self.tag_cache):
            if tag in track_tags:
                tag_vec[idx] = 1

        features.extend(tag_vec)

        return tag_vec

    def build_track_vecs(self, track_ids: List[str]) -> List[List[float]]:
        res = []
        for tid in track_ids:
            res.append(self.build_track_vec(tid))

        return res
    
    def get_training_split(self, split: float = 0.8) -> List[str]:
        return self.db.get_track_id_paritions([split])
