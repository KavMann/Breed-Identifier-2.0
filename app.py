from __future__ import annotations

from flask import Flask, jsonify, make_response, render_template, request

from inference import InferenceService, validate_request_size
from predict import MAX_IMAGE_SIZE_BYTES

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE_BYTES
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

inference_service = InferenceService()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/")
def index():
    return make_response(render_template("index.html"))


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict_upload():
    if request.method == "OPTIONS":
        return "", 204

    try:
        validate_request_size(request)
        file = request.files.get("file")
        if file is None:
            raise ValueError("Please choose an image file.")

        result = inference_service.predict_uploaded_file(file)
        return jsonify({"ok": True, "result": result})
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 400


@app.route("/predict-url", methods=["POST", "OPTIONS"])
def predict_url():
    if request.method == "OPTIONS":
        return "", 204

    try:
        image_url = request.form.get("image_url", "")
        result = inference_service.predict_url(image_url)
        return jsonify({"ok": True, "result": result})
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 400


@app.errorhandler(413)
def handle_large_upload(_error):
    max_mb = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
    return (
        jsonify(
            {
                "ok": False,
                "error": f"Uploaded image must be {max_mb:.0f} MB or smaller.",
            }
        ),
        413,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
