import sqlite3
import os
import shutil

PATH_LASTFM_DB = "data/lastfm.db"
PATH_MSD_DB = "data/msd.db"
PATH_COMBINED_DB = "data/combined.db"

def main():
    if not os.path.exists(PATH_LASTFM_DB):
        raise FileNotFoundError(f"Could not find {PATH_LASTFM_DB}")
    if not os.path.exists(PATH_MSD_DB):
        raise FileNotFoundError(f"Could not find {PATH_MSD_DB}")

    # Create a copy of lastfm.db -> combined.db
    if os.path.exists(PATH_COMBINED_DB):
        while True:
            response = input("combined.db already exists. Overwrite it? (y/n) ")
            if response.lower() == "y":
                os.remove(PATH_COMBINED_DB)
                break
            elif response.lower() == "n":
                print("Aborting.")
                return
            else:
                print("Invalid response. Please enter 'y' or 'n'")

    shutil.copyfile(PATH_LASTFM_DB, PATH_COMBINED_DB)

    # Connect to the combined db
    conn = sqlite3.connect(PATH_COMBINED_DB)
    c = conn.cursor()

    # Attach the msd database
    c.execute(f"ATTACH DATABASE '{PATH_MSD_DB}' AS msd")

    # Add new columns to the tracks table if they don't already exist
    new_columns = [
        ("danceability", "REAL"),
        ("duration", "REAL"),
        ("energy", "REAL"),
        ("key", "INTEGER"),
        ("loudness", "REAL"),
        ("mode", "INTEGER"),
        ("tempo", "REAL"),
        ("time_signature", "INTEGER"),
        ("year", "INTEGER")
    ]

    c.execute("PRAGMA table_info(tracks)")
    existing_cols = [row[1] for row in c.fetchall()]

    for col_name, col_type in new_columns:
        if col_name not in existing_cols:
            c.execute(f"ALTER TABLE tracks ADD COLUMN {col_name} {col_type}")

    # Update the tracks with MSD data
    c.execute("""
              WITH msd_data AS (
    SELECT track_id, danceability, duration, energy, key, loudness, mode, tempo, time_signature, year
    FROM msd.tracks
    )
    UPDATE tracks
    SET
    danceability = (SELECT msd_data.danceability FROM msd_data WHERE msd_data.track_id = tracks.track_id),
    duration = (SELECT msd_data.duration FROM msd_data WHERE msd_data.track_id = tracks.track_id),
    energy = (SELECT msd_data.energy FROM msd_data WHERE msd_data.track_id = tracks.track_id),
    key = (SELECT msd_data.key FROM msd_data WHERE msd_data.track_id = tracks.track_id),
    loudness = (SELECT msd_data.loudness FROM msd_data WHERE msd_data.track_id = tracks.track_id),
    mode = (SELECT msd_data.mode FROM msd_data WHERE msd_data.track_id = tracks.track_id),
    tempo = (SELECT msd_data.tempo FROM msd_data WHERE msd_data.track_id = tracks.track_id),
    time_signature = (SELECT msd_data.time_signature FROM msd_data WHERE msd_data.track_id = tracks.track_id),
    year = (SELECT msd_data.year FROM msd_data WHERE msd_data.track_id = tracks.track_id);
    """)


    # make sure to remove the tracks in the similarity table that are not in the tracks table
    c.execute("DELETE FROM track_similars WHERE track_a NOT IN (SELECT track_id FROM tracks)")
    c.execute("DELETE FROM track_similars WHERE track_b NOT IN (SELECT track_id FROM tracks)")

    # do the same with the track_tags table
    c.execute("DELETE FROM track_tags WHERE track_id NOT IN (SELECT track_id FROM tracks)")

    # now delete the tags that are not in the track_tags table
    c.execute("DELETE FROM tags WHERE tag_id NOT IN (SELECT tag_id FROM track_tags)")

    # now delete the tracks that don't show up in the track_similars table
    c.execute("DELETE FROM tracks WHERE track_id NOT IN (SELECT track_a FROM track_similars) or track_id NOT IN (SELECT track_b FROM track_similars)")


    # delete anything from the tracks table that has null values
    c.execute("DELETE FROM tracks WHERE danceability IS NULL")


    conn.commit()
    conn.close()

    print("Successfully created combined.db with MSD data merged into the tracks table.")

if __name__ == "__main__":
    main()
