#!/usr/bin/env python3
"""
app.py — Web interface for clean_video.py
Run: python app.py  then open http://localhost:5000
"""

import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mov", "m4v"}

# job_id -> {"status": ..., "log": [...], "output": ...}
jobs: dict[str, dict] = {}
job_queues: dict[str, queue.Queue] = {}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["video"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type. Use MP4 or MOV."}), 400

    job_id = str(uuid.uuid4())[:8]
    filename = secure_filename(f.filename)
    input_path = UPLOAD_DIR / f"{job_id}_{filename}"
    f.save(str(input_path))

    return jsonify({"job_id": job_id, "filename": filename, "input_path": str(input_path)})


@app.route("/process", methods=["POST"])
def process():
    data = request.json
    job_id = data.get("job_id")
    input_path = data.get("input_path")

    if not job_id or not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Invalid job or missing file"}), 400

    stem = Path(input_path).stem
    output_path = str(OUTPUT_DIR / f"{stem}_cleaned.mp4")

    # Build command
    cmd = [sys.executable, "clean_video.py", "--input", input_path]

    model = data.get("model", "base")
    cmd += ["--model", model]

    transition = data.get("transition", "crossfade")
    cmd += ["--transition", transition]

    if data.get("intro_title"):
        cmd += ["--intro-title", data["intro_title"]]
    if data.get("intro_subtitle"):
        cmd += ["--intro-subtitle", data["intro_subtitle"]]
    if data.get("intro_color"):
        cmd += ["--intro-color", data["intro_color"].lstrip("#")]
    if data.get("intro_duration"):
        cmd += ["--intro-duration", str(data["intro_duration"])]

    if data.get("outro_text"):
        cmd += ["--outro-text", data["outro_text"]]
    if data.get("outro_color"):
        cmd += ["--outro-color", data["outro_color"].lstrip("#")]
    if data.get("outro_duration"):
        cmd += ["--outro-duration", str(data["outro_duration"])]

    if data.get("lower_third_name"):
        cmd += ["--lower-third-name", data["lower_third_name"]]
    if data.get("lower_third_title"):
        cmd += ["--lower-third-title", data["lower_third_title"]]
    if data.get("lower_third_at"):
        cmd += ["--lower-third-at", str(data["lower_third_at"])]

    jobs[job_id] = {"status": "running", "log": [], "output": output_path}
    q = queue.Queue()
    job_queues[job_id] = q

    def run():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip()
                jobs[job_id]["log"].append(line)
                q.put(line)
            proc.wait()
            if proc.returncode == 0:
                jobs[job_id]["status"] = "done"
                q.put("__DONE__")
            else:
                jobs[job_id]["status"] = "error"
                q.put("__ERROR__")
        except Exception as e:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["log"].append(str(e))
            q.put(f"ERROR: {e}")
            q.put("__ERROR__")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id):
    """Server-Sent Events — streams log lines to the browser in real time."""
    def generate():
        q = job_queues.get(job_id)
        if not q:
            yield "data: Job not found\n\n"
            return
        while True:
            try:
                line = q.get(timeout=60)
                if line in ("__DONE__", "__ERROR__"):
                    yield f"data: {line}\n\n"
                    break
                yield f"data: {line}\n\n"
            except queue.Empty:
                yield "data: \n\n"  # heartbeat

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    output_path = Path(job["output"])
    return send_from_directory(output_path.parent, output_path.name, as_attachment=True)


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify({"status": job["status"], "log": job["log"]})


if __name__ == "__main__":
    print("\n  Video Editor UI → http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
