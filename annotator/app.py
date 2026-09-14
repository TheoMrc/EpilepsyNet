"""
Flask setup for the annotator app EFAS (Epileptic Fish Annotation System)
"""

import os
import random
import secrets

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

from epilepsy_net.inference import predict_video_timestamps
from epilepsy_net.utils import (
    DATA_DIR,
    get_all_videos,
    get_annotations_to_review,
    get_annotations_to_review_start_only,
    get_leaderboard_data,
    get_unannotated_videos,
    load_all_annotations,
    load_experiment_annotations,
    load_users,
    save_annotations,
)

app = Flask(__name__)
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", "0") == "1"
app.secret_key = os.environ.get("EPILEPSYNET_SECRET_KEY", secrets.token_hex(32))


VIDEO_DIRECTORY = os.path.abspath(DATA_DIR)
EXAMPLE_DIRECTORY = os.path.join(str(app.static_folder), "example_videos")
REVIEWER_USERS = {
    name.strip()
    for name in os.environ.get(
        "EPILEPSYNET_REVIEWERS", "Cool Zebrafish,Motivated Shark"
    ).split(",")
    if name.strip()
}


@app.context_processor
def inject_cookies():
    """Injects the cookies into the template context"""
    return {
        "username": session.get("username"),
        "chosen_experiment": session.get("chosen_experiment"),
    }


@app.route("/")
def home():
    """Returns the main page of the app"""
    all_videos = get_all_videos()
    all_annotations = load_all_annotations()
    unannotated_videos = get_unannotated_videos(all_videos, all_annotations)
    example_videos = (
        [f for f in os.listdir(EXAMPLE_DIRECTORY) if f.endswith(".webm")]
        if os.path.isdir(EXAMPLE_DIRECTORY)
        else []
    )
    nb_videos = sum(
        len(videos)
        for conditions in all_videos.values()
        for videos in conditions.values()
    )
    nb_unannotated = len(unannotated_videos)
    progress = (nb_videos - nb_unannotated) / nb_videos if nb_videos > 0 else 0
    users = load_users()
    return render_template(
        "home.html",
        nb_videos=nb_videos,
        nb_unannotated=nb_unannotated,
        example_videos=example_videos,
        progress=progress,
        users=users,
        can_review=session.get("username") in REVIEWER_USERS,
    )


@app.route("/experiment_choice")
def choose_experiment():
    """Allows to choose the experiment to annotate"""
    all_videos = get_all_videos()
    all_annotations = load_all_annotations()
    progress_per_experiment = {}
    for experiment in all_videos:
        exp_videos = {experiment: all_videos[experiment]}
        nb_videos = sum(
            len(videos)
            for conditions in exp_videos.values()
            for videos in conditions.values()
        )
        unannotated_videos = get_unannotated_videos(exp_videos, all_annotations)
        nb_unannotated = len(unannotated_videos)
        progress = (nb_videos - nb_unannotated) / nb_videos if nb_videos > 0 else 0
        progress_per_experiment[experiment] = progress
    return render_template(
        "choose_experiment.html",
        progress_per_experiment=progress_per_experiment,
    )


@app.route("/store_experiment_in_session", methods=["POST"])
def store_experiment_in_session():
    """Store the experiment in the session"""
    session["chosen_experiment"] = request.form.get("chosen_experiment")
    return redirect(url_for("load_unannotated_video"))


@app.route("/store_username_in_session", methods=["POST"])
def store_username_in_session():
    """Store the username in the session"""
    session["username"] = request.form.get("username")
    return redirect(url_for("home"))


@app.route("/videos")
def list_videos():
    """Allows to see all the uploaded videos for the chosen experiment"""
    chosen_experiment = session.get("chosen_experiment")
    videos = get_all_videos()
    if chosen_experiment != "all":
        videos = {chosen_experiment: videos[chosen_experiment]}
    return render_template(
        "videos.html",
        videos=videos,
    )


@app.route("/continue_annotation")
def load_unannotated_video():
    """Loads a random unannotated video for the chosen experiment(s)"""
    chosen_experiment = session.get("chosen_experiment")
    videos = get_all_videos()
    annotations = load_all_annotations()
    if chosen_experiment != "all":
        videos = {chosen_experiment: videos[chosen_experiment]}
    unannotated_videos = get_unannotated_videos(videos, annotations)
    if not unannotated_videos:
        return render_template("congrats.html")
    experiment, condition, video_name = random.choice(unannotated_videos)
    video_number = video_name.removeprefix("fish_").removesuffix(".webm")
    return redirect(f"/annotate/{experiment}/{condition}/{video_number}")


@app.route("/annotate/<experiment>/<condition>/<video_number>")
def annotate_video(experiment, condition, video_number):
    """Returns the annotation page of the app with the current video"""
    annotations = load_experiment_annotations(experiment)
    video_annotations = annotations.get(condition, {}).get(
        f"fish_{video_number}.webm", {}
    )
    video_timestamps = video_annotations.get("timestamps", [])
    video_comment = video_annotations.get("comment", "")
    return render_template(
        "annotate.html",
        experiment=experiment,
        condition=condition,
        video_number=video_number,
        video_timestamps=video_timestamps,
        video_comment=video_comment,
    )


@app.route("/leaderboard")
def leaderboard():
    """Returns the leaderboard page of the app"""
    all_annotations = load_all_annotations()
    users = load_users()
    leaderboard_data = get_leaderboard_data(all_annotations, users)
    return render_template(
        "leaderboard.html",
        leaderboard_data=leaderboard_data,
    )


@app.route("/predict/<experiment>/<condition>/<video_number>")
def predict_timestamps(experiment: str, condition: str, video_number: str):
    """
    Return model’s auto‐annotations for this video as JSON.
    """
    preds: list[dict] = predict_video_timestamps(experiment, condition, video_number)
    # preds is a list of {state, start, end}
    return jsonify(preds)


@app.route("/save_timestamps", methods=["POST"])
def save_timestamps():
    """Save the annotations to the JSON file"""
    request_data = request.json
    experiment = request_data.get("experiment")
    condition = request_data.get("condition")
    video_number = request_data.get("video_number")
    video_name = f"fish_{video_number}.webm"
    timestamps = request_data.get("timestamps")
    user_comment = request_data.get("user_comment")
    username = session.get("username")
    exp_annotations = load_experiment_annotations(experiment)
    if condition not in exp_annotations:
        exp_annotations[condition] = {}
    exp_annotations[condition][video_name] = {
        "timestamps": timestamps,
        "comment": user_comment,
        "username": username,
    }
    save_annotations(exp_annotations, experiment)
    return jsonify({"status": "success", "message": "Timestamps saved successfully"})


@app.route("/serve_video/<experiment>/<condition>/<video_number>")
def serve_video(experiment, condition, video_number):
    """Serves a video file from outside the app folder"""
    video_path = f"{experiment}/{condition}/fish_videos/fish_{video_number}.webm"
    return send_from_directory(directory=VIDEO_DIRECTORY, path=video_path)


@app.route("/start_review", methods=["POST"])
def start_review():
    """Start review mode"""
    session["review_mode"] = True
    session["review_target"] = request.form.get("target_user")
    session["review_start_only"] = bool(request.form.get("start_only"))
    return redirect(url_for("review_next"))


@app.route("/review_next")
def review_next():
    """Get next video to review from user under review"""
    if not session.get("review_mode"):
        return redirect(url_for("home"))
    target_user = session["review_target"]
    all_annotations = load_all_annotations()
    # gather only that user's annotations
    if session.get("review_start_only"):
        annotations_to_review, total_annots_for_target_user = (
            get_annotations_to_review_start_only(all_annotations, target_user)
        )
    else:
        annotations_to_review, total_annots_for_target_user = get_annotations_to_review(
            all_annotations, target_user
        )
    reviewed_count = total_annots_for_target_user - len(annotations_to_review)

    if not annotations_to_review:
        return f"All annotations from {target_user} have been reviewed!"
    exp, cond, vid_name, data = annotations_to_review[0]
    video_number = vid_name.removeprefix("fish_").removesuffix(".webm")
    session["review_task"] = (exp, cond, vid_name)
    progress = reviewed_count / total_annots_for_target_user
    return render_template(
        "annotate.html",
        experiment=exp,
        condition=cond,
        video_number=video_number,
        video_timestamps=data.get("timestamps", []),
        video_comment=data.get("comment", ""),
        review_mode=True,
        review_target=target_user,
        reviewed_count=reviewed_count,
        total_to_review=total_annots_for_target_user,
        progress=progress,
    )


@app.route("/save_review", methods=["POST"])
def save_review():
    """Save annotation when in review mode"""
    request_data = request.json
    review_task = session.get("review_task")
    if not review_task:
        return jsonify(
            {"status": "error", "message": "No review task is currently active."}
        ), 400
    exp, cond, vid_name = review_task
    print(f"{exp=}, {cond=}, {vid_name=}")

    timestamps = request_data.get("timestamps")
    user_comment = request_data.get("user_comment")
    annotations = load_experiment_annotations(exp)
    if cond not in annotations:
        annotations[cond] = {}

    annotations[cond][vid_name].update(
        {
            "timestamps": timestamps,
            "comment": user_comment,
            "reviewed": session.get("username"),
        }
    )

    save_annotations(annotations, exp)
    return jsonify({"status": "success", "message": "Review saved successfully"})


def main():
    """Main function to run the Flask app"""
    app.run()


if __name__ == "__main__":
    main()
