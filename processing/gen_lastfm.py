# gen_lastfm.py
# Takes all of the json data from the lastfm_subset and generates a database from it
# This is because the data is very annoyingly formatted into a bunch of json files

# In the database, We should generate the following tables:
# - tags, which encompasses the tags of the tracks. These tags could be genres, moods, etc.
#   - tag primary key
#   - tag name

# - track_tags, which is a relationship table between tracks and tags
#   - track id, which is the track id assigned by msd
#   - tag id, which is the id of the tag
#   - count, which is the number of times the tag appears on the track

# - track_similars, which is a relationship table between tracks and similar tracks. These are bidirectional
#   - track_a, which is the track id of the first track
#   - track_b, which is the track id of the second track
#   - similarity, which is the similarity between the two tracks

# - tracks, which encompasses the attributes of the tracks included in the lastfm_subset
#   - id, which is the ID provided by the million song dataset
#   - title, which is the title of the track
#   - artist (foreign key), which is the artist of the track

# - artists, which encompasses the attributes of the artists included in the lastfm_subset
#   - artist_id, which is the ID provided by the million song dataset
#   - artist_name, which is the name of the artist

# The raw data is encoded in python, where the title of the file is the track id
# It contains a dictionary with the following keys
# - similars, which is a list of similar tracks (tuples of track id and similarity)
# - tags, which is a list of tags (tuples of tag name and count)

# The database should be stored in a file called "lastfm.db" located in the data directory

import sqlite3
import os
import json


OUTPUT_DB = "data/lastfm.db"
PATH_LASTFM_SUBSET_DIR = "data/lastfm_subset/"

def collect_files():
    # get all the files in the directory that start with TR, and are json files
    # we should do this recursively, as there are subdirectories
    files = []
    for root, dirs, fs in os.walk(PATH_LASTFM_SUBSET_DIR):
        for f in fs:
            if f.startswith("TR") and f.endswith(".json"):
                files.append(os.path.join(root, f))
    return files

def create_db():
    if os.path.exists(OUTPUT_DB):
        # ask the user if they want to overwrite the database
        while True:
            response = input("The database already exists. Do you want to overwrite it? (y/n) ")
            if response.lower() == "y":
                os.remove(OUTPUT_DB)
                break
            elif response.lower() == "n":
                return sqlite3.connect(OUTPUT_DB)
            else:
                print("Invalid response. Please enter 'y' or 'n'")
    
    conn = sqlite3.connect(OUTPUT_DB)
    c = conn.cursor()

    # create the tags table
    c.execute("CREATE TABLE tags (tag_id INTEGER PRIMARY KEY, tag_name TEXT)")
    c.execute("CREATE TABLE track_tags (track_id TEXT, tag_id INTEGER, count INTEGER)")

    # create the track_similars table
    c.execute("CREATE TABLE track_similars (track_a TEXT, track_b TEXT, similarity REAL)")

    # create the tracks table
    c.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY, title TEXT, artist TEXT)")

    # create the artists table
    # make the artist table id autoincrement
    c.execute("CREATE TABLE artists (artist_id INTEGER, artist_name TEXT)")

    conn.commit()
    return conn

def insert_new_tags(c, tags):
    # assert that all of the tags are unique
    tag_set = set(tags)
    # assert that the tags are all strings
    for tag in tag_set:
        assert isinstance(tag, str)

    for tag in tag_set:
        # if the tag already exists, we don't need to insert it
        c.execute("SELECT * FROM tags WHERE tag_name=?", (tag,))
        if c.fetchone() is None:
            # print(f"Inserting tag {tag}")
            c.execute("INSERT INTO tags (tag_name) VALUES (?)", (tag,))

def insert_track_tags(c, track_id, tags):
    # assert that the track_id is a string
    assert isinstance(track_id, str)
    # assert that the tags is a list
    assert isinstance(tags, list)

    for tag in tags:
        # assert that the tag is a tuple
        assert isinstance(tag, list)
        # assert that the tuple has two elements
        assert len(tag) == 2
        # assert that the first element is a string
        assert isinstance(tag[0], str)
        # assert that the second element is an integer
        # json is the string representation of the integer, let's convert it to an integer
        tag[1] = int(tag[1])
        assert isinstance(tag[1], int)

        # print(f"Inserting tag {tag[0]} for track {track_id}")

        # insert the tag
        c.execute("SELECT tag_id FROM tags WHERE tag_name=?", (tag[0],))
        tag_id = c.fetchone()[0]
        c.execute("INSERT INTO track_tags (track_id, tag_id, count) VALUES (?, ?, ?)", (track_id, tag_id, tag[1]))

def insert_similarities(c, track_a, track_b, similarity):
    # assert that the tracks are strings
    assert isinstance(track_a, str)
    assert isinstance(track_b, str)
    # assert that the similarity is a float
    assert isinstance(similarity, float)

    # only insert the similarity if it doesn't already exist
    c.execute("SELECT * FROM track_similars WHERE track_a=? AND track_b=?", (track_a, track_b))
    if c.fetchone() is not None:
        return

    # insert the similarity
    c.execute("INSERT INTO track_similars (track_a, track_b, similarity) VALUES (?, ?, ?)", (track_a, track_b, similarity))
    c.execute("INSERT INTO track_similars (track_a, track_b, similarity) VALUES (?, ?, ?)", (track_b, track_a, similarity))

def insert_similars(c, track_id, similars):
    # assert that the track_id is a string
    assert isinstance(track_id, str)
    # assert that the similars is a list
    assert isinstance(similars, list)

    for similar in similars:
        # assert that the similar is a tuple
        assert isinstance(similar, list)
        # assert that the tuple has two elements
        assert len(similar) == 2
        # assert that the first element is a string
        assert isinstance(similar[0], str)
        # assert that the second element is a float
        # json is the string representation of the float, let's convert it to a float
        similar[1] = float(similar[1])
        assert isinstance(similar[1], float)

        insert_similarities(c, track_id, similar[0], similar[1])

def insert_track(c, track_id, title, artist):
    # assert that the track_id is a string
    assert isinstance(track_id, str)
    # assert that the title is a string
    assert isinstance(title, str)
    # assert that the artist is a string
    assert isinstance(artist, int)

    # only insert the track if it doesn't already exist
    c.execute("SELECT * FROM tracks WHERE track_id=?", (track_id,))
    if c.fetchone() is not None:
        return

    # insert the track
    c.execute("INSERT INTO tracks (track_id, title, artist) VALUES (?, ?, ?)", (track_id, title, artist))

def insert_artist(c, artist_name):
    # assert that the artist_name is a string
    assert isinstance(artist_name, str)

    # only insert the artist if it doesn't already exist
    c.execute("SELECT * FROM artists WHERE artist_name=?", (artist_name,))
    if c.fetchone() is not None:
        # just return the id
        print(f"Artist {artist_name} already exists")
        c.execute("SELECT artist_id FROM artists WHERE artist_name=?", (artist_name,))
        return c.fetchone()[0]
        
    
    # get the number of artists
    c.execute("SELECT COUNT(*) FROM artists")
    artist_id = c.fetchone()[0] * 100

    c.execute("INSERT INTO artists (artist_id, artist_name) VALUES (?, ?)", (artist_id, artist_name))

    return artist_id


def add_files_to_db(c, file):
    assert os.path.exists(file)
    data = None
    with open(file, "r") as f:
        data = json.load(f)

    if data is None:
        print(f"Failed to load {file}")
        return

    tags = data["tags"]
    similars = data["similars"]
    track_id = os.path.basename(file)
    track_id = track_id[:track_id.index(".json")]

    # insert the tags
    insert_new_tags(c, [tag[0] for tag in tags])
    insert_track_tags(c, track_id, tags)
    insert_similars(c, track_id, similars)
    artist_id = insert_artist(c, data["artist"])
    insert_track(c, track_id, data["title"], artist_id)

    c.connection.commit()

def main():
    # collect all of the files in the directory
    json_files = collect_files()
    print(f"Found {len(json_files)} files")

    # json_files = json_files[:10]

    # create the database
    conn = create_db()
    c = conn.cursor()

    # get the number of artists, so we can use that as the artist id
    c.execute("SELECT COUNT(*) FROM artists")
    num_artists = c.fetchone()[0]

    # iterate through all of the files
    for idx, file in enumerate(json_files):
        add_files_to_db(c, file)

        if idx % 10 == 0 and idx != 0:
            print(f"Processed {idx} files")

        # every once in a while, commit the changes
        if idx % 100 == 0 and idx != 0:
            print("Committing changes")
            conn.commit()

if __name__ == "__main__":
    main()
