from pathlib import Path
from subprocess import check_output

from flask import request, make_response, jsonify, flash, redirect
from werkzeug.utils import secure_filename
from mlflow.server import app

from .salary_model import SalaryModel
from ..salary import consts


app.config["MAX_CONTENT_LENGTH"] = 16 * 1000 * 1000


@app.route("/<name>/predict", methods=["POST"])
def predict(name: str):
    if request.headers.get("Content-Type") == "application/json":
        model = SalaryModel(f"{consts.MODELS_DIR}/{secure_filename(name)}.onnx")
        y = model.predict(request.json)
        return jsonify({y.name: y.to_list()})

    return make_response("bad request", 400)


@app.route("/<name>/metadata", methods=["GET"])
def metadata(name: str):
    model = SalaryModel(f"{consts.MODELS_DIR}/{secure_filename(name)}.onnx")
    meta = model.get_metadata()
    return jsonify(meta)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return f"""
        <!doctype html>
        <title>Upload</title>
        <h1>Upload file: [year]{consts.RAW_FILENAME_SUFFIX}/h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        """

    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]

    if not file or file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    filename = secure_filename(file.filename)
    suffix = consts.RAW_FILENAME_SUFFIX

    if not filename.endswith(suffix) or 1990 < int(filename.removesuffix(suffix)) > 2030:
        flash("Invalid file name")
        return redirect(request.url)

    path = Path(consts.RAW_DIR).joinpath(filename)
    file.save(path)
    return redirect(request.url)


@app.route("/<name>/repro", methods=["GET"])
def repro(name: str):
    dag = f"{consts.DAGS_DIR}/{secure_filename(name)}/dvc.yaml"
    return check_output(["dvc", "repro", "-f", dag]).decode("utf-8")


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
