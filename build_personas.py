import argparse
import json
import os
from uusic.common.persona import Persona
from uusic.common.db import DBInterface

def main():
    parser = argparse.ArgumentParser(description="Build personas programmatically and save to JSON.")
    parser.add_argument("--db-path", type=str, required=True, help="Path to the SQLite database file.")
    parser.add_argument("--num-personas", type=int, default=10, help="Number of personas to generate (default: 10).")
    parser.add_argument("--num-user-history", type=int, default=20, help="Number of tracks in user history (default: 20).")
    parser.add_argument("--num-recommended", type=int, default=10, help="Number of recommended tracks (default: 5).")
    parser.add_argument("--num-not-recommended", type=int, default=10, help="Number of not recommended tracks (default: 5).")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file.")

    args = parser.parse_args()

    db = DBInterface(db_path=args.db_path)
    personas = Persona.build_persona_set_algorithmic(
        num_personas=args.num_personas,
        num_user_history=args.num_user_history,
        num_reccomended=args.num_recommended,
        num_not_reccomended=args.num_not_recommended,
        db=db
    )

    # Convert to JSON-serializable format
    personas_dict = [p.__dict__() for p in personas]

    # create the file/directory if it doesn't exist
    dir_path = os.path.dirname(args.output)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(personas_dict, f, indent=4)

    print(f"Saved {len(personas)} personas to {args.output}")

if __name__ == "__main__":
    main()
