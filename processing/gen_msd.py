# gen_msd.py
# the objective of this script is to parse through the MSD dataset and generate a database from it to work with
# there are a lot of fields in the dataset, but we are only interested in a few of them
# - track_id, which is the ID of the track
# - danceability
# - duration
# - energy
# - key
# - loudness
# - mode
# - tempo
# - time_signature
# - year

# this will be combined with the lastfm database to create a larger database, which will be used to generate recommendations


import sqlite3
import os
import json
import h5py
import pprint
import hdf5_getters
import numpy as np



OUTPUT_DB = "data/msd.db"
PATH_MSD_SUBSET_DIR = "data/msd_subset/"

def collect_files():
    # get all the files in the directory that start with TR, and are json files
    # we should do this recursively, as there are subdirectories
    files = []
    for root, dirs, fs in os.walk(PATH_MSD_SUBSET_DIR):
        for f in fs:
            if f.startswith("TR") and f.endswith(".h5"):
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

    # create the single table, tracks
    c.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY, danceability REAL, duration REAL, energy REAL, key INTEGER, loudness REAL, mode INTEGER, tempo REAL, time_signature INTEGER, year INTEGER)")

    conn.commit()
    return conn

def print_structure(f, level=0):
    for key in f.keys():
        print("  " * level + key)
        if isinstance(f[key], h5py.Group):
            print_structure(f[key], level + 1)

def add_file_to_db(c, file):
    # parse the hdf5 file
    f = hdf5_getters.open_h5_file_read(file)

    # get all the data we need
    data = [
        hdf5_getters.get_track_id(f),
        hdf5_getters.get_danceability(f),
        hdf5_getters.get_duration(f),
        hdf5_getters.get_energy(f),
        hdf5_getters.get_key(f),
        hdf5_getters.get_loudness(f),
        hdf5_getters.get_mode(f),
        hdf5_getters.get_tempo(f),
        hdf5_getters.get_time_signature(f),
        hdf5_getters.get_year(f)
    ]

    # all of the data is in the form of numpy arrays, so we need to convert them to python types
    data = [x.item() if isinstance(x, np.generic) else x for x in data]
    # for the track id, we need to convert it to a string
    # right now it is a binary string, which is not what we want
    data[0] = data[0].decode("utf-8")

    # pprint.pprint(data)

    # insert the data into the database
    c.execute("INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)

    # close the file
    f.close()


def main():
    # collect all of the files in the directory
    data_files = collect_files()
    print(f"Found {len(data_files)} files")

    # json_files = json_files[:10]

    # create the database
    conn = create_db()
    c = conn.cursor()

    # iterate through all of the files
    for idx, file in enumerate(data_files):
        add_file_to_db(c, file)

        if idx % 10 == 0 and idx != 0:
            print(f"Processed {idx} files")

        # every once in a while, commit the changes
        if idx % 100 == 0 and idx != 0:
            print("Committing changes")
            conn.commit()

if __name__ == "__main__":
    main()