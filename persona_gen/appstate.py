# appstate.py
from uusic.common.db import DBInterface
from uusic.common.persona import Persona

class AppState:
    def __init__(self):
        self.personas = []
        self.personas.append(Persona())
        self.current_persona_idx : int = 0
        self.db : DBInterface = DBInterface("data/combined.db")

        self.filter_tags = []

    @property
    def current_persona(self) -> Persona:
        return self.personas[self.current_persona_idx]
    
    @property
    def applied_tags(self):
        return [tag["name"] for tag in self.filter_tags if tag["active"]]
    
    def create_persona(self) -> Persona:
        self.current_persona_idx = len(self.personas)
        persona = Persona()
        self.personas.append(persona)
        return persona

    def set_filter_tags(self, tags):
        self.filter_tags = tags

    def add_filter_tag(self, tag):
        for t in self.filter_tags:
            if t["name"] == tag:
                t["active"] = True
                return
            
        self.filter_tags.append({"name": tag, "active": True})


    def remove_filter_tag(self, tag):
        for t in self.filter_tags:
            if t["name"] == tag:
                self.filter_tags.remove(t)
                break

    def is_filter_tag_active(self, tag):
        for t in self.filter_tags:
            if t["name"] == tag:
                print("Found tag", t)
                return t["active"]
        return False

    def get_persona(self, idx):
        if idx < 0 or idx >= len(self.personas):
            return None
        return self.personas[idx]
    
    def get_all_personas(self):
        return self.personas
    
    def get_tags(self, count):
        if not self.filter_tags or len(self.filter_tags) == 0:
            return self.get_popular_tags(count)
        else:
            return self.filter_tags
    
    def get_popular_tags(self, count):
        tag_names = self.db.get_common_tags(count)
        # zip it up with active/inactive status
        tags = [{"name": tag, "active": False in self.filter_tags} for tag in tag_names]
        self.filter_tags = tags
        return tags
    
    def get_reccomended_tracks(self, persona_id, count):
        # get songs that are similar to the last song in the persona's history
        # otherwise just grab some popular songs
        persona = self.get_persona(persona_id)

        track_ids = []
        if persona is None or len(persona.listening_history) == 0:
            track_ids = self.db.get_random_songs_by_tag(self.applied_tags, count)
        else:
            track_id_results = []
            for listened in persona.listening_history:
                results = self.db.get_similar_tracks_filtered(listened.track_id, self.applied_tags, count)
                track_id_results.append(results) # list of lists

            # flatten the list by interleaving the results
            track_ids = [item for sublist in zip(*track_id_results) for item in sublist]
            track_ids = list(set(track_ids))

        # filter the tracks if there are already in the persona's history
        persona_track_ids = [item.track_id for item in persona.listening_history]
        persona_track_ids += [item.track_id for item in persona.reccomended_tracks]
        persona_track_ids += [item.track_id for item in persona.not_reccomended_tracks]

        track_ids = [track_id for track_id in track_ids if track_id not in persona_track_ids]

        # now if we don't have enough tracks, just grab some random ones
        if len(track_ids) < count:
            additional_tracks = self.db.get_random_songs_by_tag(self.applied_tags, count - len(track_ids))
            track_ids += additional_tracks


        return self.db.convert_trackid_list_to_tracks(track_ids)
    
