# gen_db.py (archive)
# Purpose: Generate a single database from the individual databases, which should be easier to work with

# ! This script was ran to find the matching songs between the lastfm database and the music_genre_classification.csv file
# ! However, after running the script, it was found that there was very little overlap between the two databases
# ! Therefore, the script was not used to generate the collective database

# Most of the information is stored in lastfm.db, but we should also include the music_genre_classification.csv and music_genre_classification_classes.csv files
# We are mainly going to be focusing on exapnding the tracks table, as we have a lot of information about the tracks in the music_genre_classification.csv file

# In the database, we should have the following tables:
# - tracks, which encompasses the attributes of the tracks
#   - id, which is the ID provided by the million song dataset
#   - title, which is the title of the track
#   - artist (foreign key), which is the artist of the track
#   - popularity, which is the popularity of the track
#   - danceability, which is the danceability of the track
#   - energy, which is the energy of the track
#   - key, which is the key of the track
#   - loudness, which is the loudness of the track
#   - mode, which is the mode of the track
#   - speechiness, which is the speechiness of the track
#   - acousticness, which is the acousticness of the track

# tags, which is copied from the lastfm database (see below)
# track_tags, which is copied from the lastfm database (see below)
# track_similars, which is copied from the lastfm database (see below)
# artists, which is copied from the lastfm database (see below)

# The database should be stored in a file called "tracks.db"

# We construct the databse using a few different sources:
# - the lastfm_similars.db, which contains the similarities between tracks
#       - tags, which encompasses the tags of the tracks. These tags could be genres, moods, etc.
#           - tag_id, which is the tag id
#           - tag_name, which is the name of the tag
#       - track_tags, which is a relationship table between tracks and tags
#           - track_id, which is the track id assigned by msd
#           - tag_id, which is the id of the tag
#           - count, which is the number of times the tag appears on the track
#       - track_similars, which is a relationship table between tracks and similar tracks. These are bidirectional
#           - track_a, which is the track id of the first track
#           - track_b, which is the track id of the second track
#           - similarity, which is the similarity between the two tracks
#       - tracks, which encompasses the attributes of the tracks included in the lastfm_subset
#           - track_id, which is the ID provided by the million song dataset
#           - title, which is the title of the track
#           - artist (foreign key), which is the artist of the track
#       - artists, which encompasses the attributes of the artists included in the lastfm_subset
#           - artist_id, which is the ID provided by the million song dataset
#           - artist_name, which is the name of the artist
#   - music_genre_classification.csv, which contains attributes of the tracks
#       - artist_name, track_name, popualrity, danceability, energy, key, loudness, mode, speechiness, acousticness
#   - music_genre_classification_classes.csv, which contains the classes of the tracks
#       - boolean columns for each class: acoustic/folk, alt music, blues, bollywood, country, hip hop, indie alt, instrumental, metal, or pop
#       - these match the entries in the music_genre_classification.csv file, line by line
# 


PATH_LAST_FM_DB = "data/lastfm.db"
PATH_TRACK_META = "data/music_genre_classification.csv"
OUTPUT_DB = "data/collective.db"

import sqlite3
import os
import csv
import pprint

from difflib import SequenceMatcher

class GenreIndexes:
    ARTIST_NAME = 0
    TRACK_NAME = 1
    POPULARITY = 2
    DANCEABILITY = 3
    ENERGY = 4
    KEY = 5
    LOUDNESS = 6
    MODE = 7
    SPEECHINESS = 8
    ACOUSTICNESS = 9
    INSTRUMENTALNESS = 10
    LIVENESS = 11
    VALENCE = 12
    TEMPO = 13
    DURATION = 14
    TIME_SIGNATURE = 15


def load_source_db():
    return sqlite3.connect(PATH_LAST_FM_DB)

def create_db():
    if os.path.exists(OUTPUT_DB):
        # ask the user if they want to overwrite the database
        while True:
            response = input(f"The collective database ({OUTPUT_DB}) already exists. Do you want to overwrite it? (y/n) ")
            if response.lower() == "y":
                os.remove(OUTPUT_DB)
                break
            elif response.lower() == "n":
                return sqlite3.connect(OUTPUT_DB)
    return sqlite3.connect(OUTPUT_DB)

def load_track_metadata():
    data = None
    with open(PATH_TRACK_META, "r", encoding="utf8") as f:
        reader = csv.reader(f)
        data = list(reader)

    if not data:
        return
    
    # remove the header
    data = data[1:]
    return data

def clean_title(title):
    # Convert to lowercase
    title = title.lower()

    # there tends to be a lot of different specifiers in the title, like (feat. ...), so
    common_seperators = [
        "-", "(", "feat.", "ft.", "ft", "featuring", "with", "and", "&", "vs.", "vs",
    ]

    # now split the words by the common seperators
    for sep in common_seperators:
        title = title.split(sep)[0]

    # remove any leading or trailing whitespace
    title = title.strip()

    # now remove any non-alphanumeric characters
    title = "".join([c for c in title if c.isalnum() or c.isspace()])
    return title

def similarity_check(str1, str2):
    return SequenceMatcher(None, clean_title(str1), clean_title(str2)).ratio()


def find_matching_songs(source_db, meta_list, save_file):
    # we need to find the matching songs between the two databases
    # meta_list is just a list of lists, where each list is a row in the csv file
    # source_db is a sqlite3 database

    # check if the save file exists, and if it does, load it and return it
    # if os.path.exists(save_file):
    #     # query the user if they want to load the file
    #     while True:
    #         response = input(f"Found a file with matching songs ({save_file}). Do you want to load it? (y/n) ")
    #         if response.lower() == "y":
    #             break
    #         elif response.lower() == "n":
    #             os.remove(save_file)
    #             break

    #     if response.lower() == "y":
    #         break
    #         # load it
    #         # with open(save_file, "r") as f:
    #         #     lines = f.readlines()
    #         #     return set([tuple(line.strip().split(",")) for line in lines])

    matches = list()

    data_reconsturcted = list()
    artist_id_map = dict()
    # use the source_db to generate a list of track-artist pairs
    c = source_db.cursor()
    c.execute("SELECT * FROM tracks")
    for row in c.fetchall():
        track_id = row[0]
        track_name = row[1]
        artist_id = row[2]
        artist_name = None
        # check if the artist is in the artist_id_map
        if artist_id in artist_id_map:
            artist_name = artist_id_map[artist_id]
        else:
            # query the database for the artist name
            c.execute("SELECT * FROM artists WHERE artist_id=?", (artist_id,))
            artist_name = c.fetchone()[1]
            artist_id_map[artist_id] = artist_name

        if artist_name == None:
            print(f"Could not find the artist name for {track_name}")
            continue

        data_reconsturcted.append([track_id, track_name, artist_name])

    # write this to a file
    with open("track_artist_pairs.txt", "w", encoding="utf8") as f:
        for data_row in data_reconsturcted:
            f.write(f"{data_row[0]},{data_row[1]}\n")



    current_row = 0

    closest_artist_similarities = dict()

    for row in meta_list:
        current_row += 1
        artist_name = row[GenreIndexes.ARTIST_NAME]
        track_name = row[GenreIndexes.TRACK_NAME]
        found_artist_match = False

        best_artist_similarity = 0
        best_artist = None

        best_track_similarity = 0
        best_track = None
        best_track_id = None

        # check if we are an artist that is in the closest_artist_similarities list
        # this means we have already checked this artist, and we should skip it
        if artist_name in closest_artist_similarities:
            print(f"Skipping artist {artist_name}, already checked for their existence")
            continue

        # now check if the track-artist pair is in the track_artist_pairs list
        # we should be really lenient with the matching, as the names might not match exactly
        for data_row in data_reconsturcted:
            # generally, we should check that that the artist name is the same, then check the track name
            # this is because the artist name is more likely to be the same than the track name
            artist_similarity = similarity_check(artist_name, data_row[2])
            if artist_similarity > best_artist_similarity:
                best_artist_similarity = artist_similarity
                best_artist = data_row[2]            

            if artist_similarity > 0.8:
                found_artist_match = True
                # now check the track name
                track_similarity = similarity_check(track_name, data_row[1])
                if track_similarity > best_track_similarity:
                    best_track_similarity = track_similarity
                    best_track = data_row[1]
                    best_track_id = data_row[0]

                if track_similarity > 0.8:
                    # we have a match
                    break

            
        if found_artist_match and best_track_similarity > 0.3: # a potential match, add it to the list to check later
            # we have a match
            matches.append([
                artist_name, track_name,
                best_artist, best_track, best_track_id,
                best_track_similarity
            ])

        # if we didn't find an artist match, we should store the best artist similarity and the artist name
        # that way if something is seriously wrong, we can check the artist name
        if not found_artist_match:
            closest_artist_similarities[artist_name] = (best_artist, best_artist_similarity)

        if current_row % 10 == 0:
            print(f"Match percentage, out of all checks: {(len(matches)/current_row)*100:.2f}%. Amount of matches: {len(matches)}, current row: {current_row}")
            print("Unmatched artists count: ", len(closest_artist_similarities))

    # print the closest artist similarities, ranked by similarity
    print("Closest artist similarities:")
    for artist, (closest_artist, similarity) in sorted(closest_artist_similarities.items(), key=lambda x: x[1][1], reverse=True):
        print(f"Artist: {artist}, Closest artist: {closest_artist}, Similarity: {similarity}")

    pprint.pprint(matches)

    potential_matches = len(meta_list)
    actual_matches = len(matches)
    print(f"Found {actual_matches} matches out of {potential_matches} potential matches ({(actual_matches/potential_matches)*100:.2f}%)")

    # save the matches to a file
    with open(save_file, "w", encoding="utf8") as f:
        for match in matches:
            f.write(",".join([str(x) for x in match]) + "\n")

    return matches

def find_meta_row(meta_list, predicate):
    for row in meta_list:
        if predicate(row):
            return row
    return None

def fine_meta_rows(meta_list, predicate):
    rows = []
    for row in meta_list:
        if predicate(row):
            rows.append(row)

    return rows

def build_meta_table(lastfm_db, meta_list, dest_db, matches):
    for artist, track in matches:
        # find the track in the metadata list
        meta_row = find_meta_row(meta_list, lambda row: row[GenreIndexes.ARTIST_NAME] == artist and row[GenreIndexes.TRACK_NAME] == track)
        if not meta_row:
            print(f"Could not find the metadata for {track} by {artist}")
            continue

        # insert the track into the database
        dest_c = dest_db.cursor()
        dest_c.execute(
            "INSERT INTO tracks (title, artist, popularity, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, time_signature) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            meta_row[GenreIndexes.POPULARITY], meta_row[GenreIndexes.DANCEABILITY], meta_row[GenreIndexes.ENERGY], meta_row[GenreIndexes.KEY], meta_row[GenreIndexes.LOUDNESS], 
            meta_row[GenreIndexes.MODE], meta_row[GenreIndexes.SPEECHINESS], meta_row[GenreIndexes.ACOUSTICNESS], meta_row[GenreIndexes.INSTRUMENTALNESS], meta_row[GenreIndexes.LIVENESS], 
            meta_row[GenreIndexes.VALENCE], meta_row[GenreIndexes.TEMPO], meta_row[GenreIndexes.DURATION], meta_row[GenreIndexes.TIME_SIGNATURE])
        
        # get the id of the track
        lastfm_c = lastfm_db.cursor()
        lastfm_c.execute("SELECT track_id FROM tracks WHERE title=? AND artist=?", (track, artist))
        track_id = lastfm_c.fetchone()[0]
    

def main():
    lastfm_db = load_source_db() # lastfm.db
    if not lastfm_db:
        print("Could not load the lastfm database")
        return
    dest_db = create_db() # collective.db
    if not dest_db:
        print("Could not create the destination database")
        return
    meta_list = load_track_metadata()
    if not meta_list:
        print("Could not load the track metadata")
        return
    
    matching_songs = find_matching_songs(lastfm_db, meta_list, "matching_songs.txt")

if __name__ == '__main__':
    main()