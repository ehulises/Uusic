from __future__ import annotations
import sqlite3
import os
from typing import List, Tuple, Dict, Optional, Any

class DBInterface:
    """
    A class-based interface to the combined SQLite database that contains
    track and artist data, tags, and similar tracks.

    This class provides a set of high-level getters, similar to the hdf5_getters
    approach, to facilitate easy data retrieval for machine learning and data analysis.
    """

    def __init__(self, db_path: str = "data/combined.db") -> None:
        """
        Initialize the DBInterface with the path to the SQLite database.

        :param db_path: Path to the SQLite database file.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at {db_path}")
        self.db_path = db_path

    def _get_db_connection(self) -> sqlite3.Connection:
        """
        Create and return a new database connection.

        :return: A connection object to the SQLite database.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_all_track_ids(self) -> List[str]:
        """
        Retrieve all track IDs from the database.

        :return: A list of all track_ids as strings.
        """
        query = "SELECT track_id FROM tracks"
        with self._get_db_connection() as conn:
            rows = conn.execute(query).fetchall()
        return [row["track_id"] for row in rows]

    def get_track_info(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a dictionary of track information for a given track_id.

        Fields:
        {
          "track_id": str,
          "title": str,
          "artist_id": int,
          "artist_name": str,
          "danceability": float,
          "duration": float,
          "energy": float,
          "key": int,
          "loudness": float,
          "mode": int,
          "tempo": float,
          "time_signature": int,
          "year": int
        }

        :param track_id: The track ID of the desired track.
        :return: A dictionary with track info, or None if track_id not found.
        """
        query = """
        SELECT t.track_id, t.title, a.artist_id, a.artist_name,
               t.danceability, t.duration, t.energy, t.key, t.loudness,
               t.mode, t.tempo, t.time_signature, t.year
        FROM tracks t
        LEFT JOIN artists a ON t.artist = a.artist_id
        WHERE t.track_id = ?
        """
        with self._get_db_connection() as conn:
            row = conn.execute(query, (track_id,)).fetchone()

        if row is None:
            return None
        return dict(row)

    def get_track_title(self, track_id: str) -> Optional[str]:
        """
        Retrieve the title of the given track_id.

        :param track_id: The track ID.
        :return: The track title, or None if not found.
        """
        info = self.get_track_info(track_id)
        return info["title"] if info else None

    def get_artist_name_by_track(self, track_id: str) -> Optional[str]:
        """
        Retrieve the artist name for the given track_id.

        :param track_id: The track ID.
        :return: The artist name, or None if not found.
        """
        info = self.get_track_info(track_id)
        return info["artist_name"] if info else None

    def get_artist_name(self, artist_id: int) -> Optional[str]:
        """
        Retrieve the artist name for the given artist_id.

        :param artist_id: The artist ID.
        :return: The artist name, or None if not found.
        """
        query = "SELECT artist_name FROM artists WHERE artist_id = ?"
        with self._get_db_connection() as conn:
            row = conn.execute(query, (artist_id,)).fetchone()
        return row["artist_name"] if row else None

    def get_track_tags(self, track_id: str) -> List[Tuple[str, int]]:
        """
        Retrieve the tags (name, count) associated with a given track_id.

        :param track_id: The track ID.
        :return: A list of tuples (tag_name, count).
        """
        query = """
        SELECT tg.tag_name, tt.count
        FROM track_tags tt
        JOIN tags tg ON tt.tag_id = tg.tag_id
        WHERE tt.track_id = ?
        ORDER BY tt.count DESC
        """
        with self._get_db_connection() as conn:
            rows = conn.execute(query, (track_id,)).fetchall()
        return [(r["tag_name"], r["count"]) for r in rows]

    def get_similar_tracks(self, track_id: str) -> List[Tuple[str, float]]:
        """
        Retrieve tracks similar to the given track_id, sorted by similarity descending.

        :param track_id: The track ID.
        :return: A list of tuples (similar_track_id, similarity).
        """
        query = """
        SELECT track_b AS similar_track_id, similarity
        FROM track_similars
        WHERE track_a = ?
        ORDER BY similarity DESC
        """
        with self._get_db_connection() as conn:
            rows = conn.execute(query, (track_id,)).fetchall()
        return [(r["similar_track_id"], r["similarity"]) for r in rows]

    def get_tracks_by_year(self, year: int) -> List[str]:
        """
        Retrieve track_ids for tracks released in a specific year.

        :param year: The year to filter tracks by.
        :return: A list of track_ids released in that year.
        """
        query = "SELECT track_id FROM tracks WHERE year = ?"
        with self._get_db_connection() as conn:
            rows = conn.execute(query, (year,)).fetchall()
        return [r["track_id"] for r in rows]

    def get_tracks_by_tempo_range(self, min_tempo: float, max_tempo: float) -> List[str]:
        """
        Retrieve track_ids whose tempo is between min_tempo and max_tempo.

        :param min_tempo: The minimum tempo.
        :param max_tempo: The maximum tempo.
        :return: A list of track_ids within the tempo range.
        """
        query = "SELECT track_id FROM tracks WHERE tempo BETWEEN ? AND ?"
        with self._get_db_connection() as conn:
            rows = conn.execute(query, (min_tempo, max_tempo)).fetchall()
        return [r["track_id"] for r in rows]

    def get_tracks_by_tag(self, tag_name: str) -> List[str]:
        """
        Retrieve track_ids that have a given tag.

        :param tag_name: The name of the tag.
        :return: A list of track_ids that have this tag.
        """
        query = """
        SELECT tt.track_id
        FROM track_tags tt
        JOIN tags t ON tt.tag_id = t.tag_id
        WHERE t.tag_name = ?
        """
        with self._get_db_connection() as conn:
            rows = conn.execute(query, (tag_name,)).fetchall()
        return [r["track_id"] for r in rows]
    
    def get_common_tags(self, number_tags) -> List[str]:
        """
        Retrieve the most common tags in the database.

        :param number_tags: The number of tags to retrieve.
        :return: A list of the most common tags.
        """
        query = """
        SELECT tag_name, COUNT(*) as count
        FROM track_tags
        JOIN tags ON track_tags.tag_id = tags.tag_id
        GROUP BY tag_name
        ORDER BY count DESC
        LIMIT ?
        """
        with self._get_db_connection() as conn:
            rows = conn.execute(query, (number_tags,)).fetchall()
        return [r["tag_name"] for r in rows]
    

    def get_similar_tracks_filtered(self, track_id : str, filter_tags : List[str], count : int) -> List[str]:
        """
        Retrieve tracks similar to the given track_id, sorted by similarity descending, with tags filtered.

        :param track_id: The track ID.
        :param filter_tags: A list of tags to filter by.
        :param count: The number of similar tracks to retrieve.
        :return: A list of track_ids of similar tracks.
        """
        query = """
        SELECT track_b AS similar_track_id, similarity
        FROM track_similars
        WHERE track_a = ?
        ORDER BY similarity DESC
        """
        with self._get_db_connection() as conn:
            rows = conn.execute(query, (track_id,)).fetchall()

        if len(filter_tags) == 0:
            return [r["similar_track_id"] for r in rows][:count]

        # now filter the rows by the tags, until we have count rows
        filtered_rows = []
        for r in rows:
            if len(filtered_rows) >= count:
                break
            # get the tags for this track
            tags = self.get_track_tags(r["similar_track_id"])
            # check if any of the tags are in the filter_tags
            if any(tag[0] in filter_tags for tag in tags):
                filtered_rows.append(r)

        return [r["similar_track_id"] for r in filtered_rows]
    
    def get_random_songs_by_tag(self, tag_names : List[str], count : int) -> List[str]:
        """
        Retrieve a list of random songs that have at least one of the given tags.

        :param tag_names: A list of tag names.
        :param count: The number of songs to retrieve.
        :return: A list of track_ids of random songs.
        """
        if len(tag_names) == 0:
            # just get random songs
            query = """
            SELECT track_id
            FROM tracks
            ORDER BY RANDOM()
            LIMIT ?
            """
            with self._get_db_connection() as conn:
                rows = conn.execute(query, [count]).fetchall()
            return [r["track_id"] for r in rows]
        else:
            query = """
            SELECT track_id
            FROM track_tags
            JOIN tags ON track_tags.tag_id = tags.tag_id
            WHERE tags.tag_name IN ({})
            ORDER BY RANDOM()
            LIMIT ?
            """.format(", ".join(["?"] * len(tag_names)))
            with self._get_db_connection() as conn:
                rows = conn.execute(query, tag_names + [count]).fetchall()
            return [r["track_id"] for r in rows]
    

    def convert_trackid_list_to_tracks(self, track_ids: List[str], include_tags : bool = True) -> List[Dict[str, Any]]:
        """
        Convert a list of track_ids to a list of track dictionaries.

        :param track_ids: A list of track IDs.
        :return: A list of dictionaries with track information.
        """
        tracks = []
        for track_id in track_ids:
            track_info = self.get_track_info(track_id)
            if track_info:
                tracks.append(track_info)

                if include_tags:
                    tags = self.get_track_tags(track_id)
                    tracks[-1]["tags"] = [{"name": tag[0], "count": tag[1]} for tag in tags]
        return tracks
    
    def get_track_tags(self, track_id : str):
        """
        Grabs all the tags for a given track

        :param track_id: The Track's ID
        """

        # you have to perform a join on the track_tags and tags table
        query = """
        SELECT tag_name, count
        FROM track_tags
        JOIN tags ON track_tags.tag_id = tags.tag_id
        WHERE track_id = ?
        """

        with self._get_db_connection() as conn:
            rows = conn.execute(query, (track_id, )).fetchall()
        return [r["tag_name"] for r in rows]
    
    def select_random_tracks_percentage(self, percentage : float):
        """
        Returns a random subset of track ids

        :param percentage: The percentage of all tracks you want to get, 0-1
        """
        count = int(percentage * len(self.get_all_track_ids()))
        return self.select_random_tracks_count(count)
    
    def select_random_tracks_count(self, count : int):
        """
        Returns a random subset of track ids

        :param count: The number of tracks you want to get
        """
        query = """
        SELECT track_id
        FROM tracks
        ORDER BY RANDOM()
        LIMIT ?
        """
        with self._get_db_connection() as conn:
            rows = conn.execute(query, (count,)).fetchall()
        return [r["track_id"] for r in rows]

    def get_track_id_paritions(self, partitions : List[float]) -> List[List[str]]:
        """
        Returns a list of track ids partitioned by the given percentages

        :param partitions: A list of percentages, 0-1
        """
        # get all of the track ids, in a random order
        # then split them up into arrays
        track_ids = self.select_random_tracks_percentage(1)
        partitioned = []
        prev_p = 0
        for p in partitions:
            new_p = prev_p + p
            start_idx = int(len(track_ids) * prev_p)
            end_idx = int(len(track_ids) * new_p)
            partitioned.append(track_ids[start_idx:end_idx])
            prev_p = new_p

        if prev_p < 1:
            partitioned.append(track_ids[end_idx:])

        return partitioned
    
    def get_similarity(self, track_a : str, track_b : str):
        """
        Returns the similarity between two tracks

        :param track_a: The first track's ID
        :param track_b: The second track's ID
        """
        query = """
        SELECT similarity
        FROM track_similars
        WHERE track_a = ? AND track_b = ?
        """
        with self._get_db_connection() as conn:
            row = conn.execute(query, (track_a, track_b)).fetchone()
        return row["similarity"] if row else None
    

    def get_similar_tracks_from_list(self, track_ids : List[str], threshold : float = 0.5) -> List[str]:
        """
        Returns a list of similar tracks to the given track ids

        :param track_ids: A list of track ids
        :param min_count: The minimum number of similar tracks to return
        :param threshold: The minimum similarity to return
        """
        # Get similar tracks based on the given track ids
        query = """
        SELECT track_b, similarity
        FROM track_similars
        WHERE track_a IN ({})
        AND similarity >= ?
        ORDER BY similarity DESC
        """.format(", ".join(["?"] * len(track_ids)))
        with self._get_db_connection() as conn:
            rows = conn.execute(query, (*track_ids, threshold)).fetchall()

        return list(set([r["track_b"] for r in rows]))