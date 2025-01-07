from flask import (
    Flask, render_template, request, jsonify, send_file
)

from uusic.common.persona import (
    ListeningHistoryItem,
    Persona
)
from persona_gen.appstate import AppState

import json
import io
from pprint import pprint

app = Flask(__name__)

appstate = AppState()


@app.route("/")
def index():
    # Show a link to create a persona
    # reroute to the persona creation page
    return create_persona()

@app.route("/personas")
def create_persona():
    # this is a dashboard for creating personas
    context = {
        "tags": appstate.get_tags(20),
        "recommended_tracks": None,
        "personas": appstate.get_all_personas()
    }

    pprint(context)

    return render_template("home.html", **context)


@app.route("/personas/all")
def list_all_personas():
    return jsonify(appstate.get_all_personas())

@app.route("/personas/<int:persona_id>")
def list_persona(persona_id: int):
    return jsonify(appstate.get_persona(persona_id))

@app.route("/personas/preview_export")
def preview_export_personas():
    return jsonify(appstate.get_all_personas())

@app.route("/tags/popular/<int:count>")
def popular_tags(count: int):
    return jsonify(appstate.get_popular_tags(count))

@app.route("/tags/add_tag", methods=["POST"])
def add_tag():
    tag = request.json.get("tag")
    appstate.add_filter_tag(tag)
    # return an okay response
    return jsonify({"status": "ok"})

@app.route("/tags/remove_tag", methods=["POST"])
def remove_tag():
    tag = request.json.get("tag")
    appstate.remove_filter_tag(tag)
    # return an okay response
    return jsonify({"status": "ok"})


@app.route("/tracks/search")
def search_tracks():
    # uses the current state of the app to search for tracks
    # based on the current persona's history
    print("Searching for tracks")

    context = {
        "tracks": appstate.get_reccomended_tracks(appstate.current_persona_idx, 20)
    }

    return render_template("snippets/tracks.html", **context)

@app.route("/tracks/render/<string:track_id>")
def render_track(track_id):
    track_info = appstate.db.get_track_info(track_id)
    context = {
        "track": track_info
    }
    return render_template("snippets/track.html", **context)


@app.route("/tracks/add_modal/<string:track_id>")
def add_track_modal(track_id):
    # check if this track is already in the persona's history, likes, or dislikes
    persona = appstate.current_persona
    valid_to_add = track_id not in [item.track_id for item in persona.listening_history]
    valid_to_add &= track_id not in [item.track_id for item in persona.reccomended_tracks]
    valid_to_add &= track_id not in [item.track_id for item in persona.not_reccomended_tracks]

    if not valid_to_add:
        # lets return an error message
        return render_template("snippets/add_track_modal_error.html")
    
    # get the track info
    track_info = appstate.db.get_track_info(track_id)
    context = {
        "track": track_info
    }
    return render_template("snippets/add_track_modal.html", **context)


@app.route("/tracks/modal/add_to_user_history/<string:track_id>", methods=["POST"])
def add_track_to_history(track_id):
    # get the raw data from the request
    data = request.data.decode("utf-8")
    print(data)
    # for some reason, htmx is sending the data as
    # key=value&key=value&key=value
    # so we need to parse it
    data = data.split("&")
    data = {key: value for key, value in [item.split("=") for item in data]}
    if "ranking" not in data:
        return "No ranking provided", 400

    ranking = data.get("ranking")
    try:
        ranking = float(ranking)
    except ValueError:
        ranking = 0

    print(f"Adding track {track_id} to history with ranking {ranking}")

    track_name = appstate.db.get_track_info(track_id).get("title")

    appstate.current_persona.add_listening_history(track_id, ranking, track_name)

    context = {
        "track_id": track_id,
        "list" : "Listening History"
    }

    return render_template("snippets/track_added.html", **context)

@app.route("/tracks/modal/add_to_reccommended/<string:track_id>", methods=["POST"])
def add_track_to_reccommended(track_id):
    # just add the track to the likes
    track_name = appstate.db.get_track_info(track_id).get("title")
    appstate.current_persona.add_reccomended_track(track_id, track_name)
    context = {
        "track_id": track_id,
        "list" : "Reccomended"
    }

    return render_template("snippets/track_added.html", **context)

@app.route("/tracks/modal/add_to_not_reccommended/<string:track_id>", methods=["POST"])
def add_track_to_not_reccommended(track_id):
    # just add the track to the dislikes
    track_name = appstate.db.get_track_info(track_id).get("title")
    appstate.current_persona.add_not_reccomended_track(track_id, track_name)
    context = {
        "track_id": track_id,
        "list" : "Not Reccomended"
    }

    return render_template("snippets/track_added.html", **context)


@app.route("/tags/flip_tag/<string:tag>", methods=["POST"])
def flip_tag(tag):
    is_active = appstate.is_filter_tag_active(tag)
    pprint(appstate.filter_tags)

    if is_active:
        print(f"Removing tag {tag}")
        appstate.remove_filter_tag(tag)
    else:
        print(f"Adding tag {tag}")
        appstate.add_filter_tag(tag)
    
    # return a render of the tag
    context = {
        "tag" : {"name": tag, "active": not is_active}
    }

    return render_template("snippets/tag.html", **context)



@app.route("/peronas/export")
def export_personas():
    # Export all personas as JSON
    data = json.dumps(appstate.get_all_personas(), indent=2)
    # Return as a downloadable file
    output = io.StringIO()
    output.write(data)
    output.seek(0)
    return send_file(
        io.BytesIO(output.read().encode('utf-8')),
        mimetype='application/json',
        as_attachment=True,
        attachment_filename="personas.json"
    )

@app.route("/personas/get")
def get_personas():
    context = {
        "personas": appstate.get_all_personas()
    }
    pprint([persona.__dict__() for persona in context["personas"]])
    return render_template("snippets/persona_list.html", **context)




if __name__ == "__main__":
    app.run(debug=True)
