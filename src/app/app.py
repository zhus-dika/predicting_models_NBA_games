from flask import Flask, request, make_response, jsonify

from .salary_pipeline import SalaryPipeline
from ..salary import consts


app = Flask(__name__)


@app.route("/forward/", methods=["POST"])
def forward():
    try:
        if request.headers.get("Content-Type") == "application/json":
            pipe = SalaryPipeline(consts.CATBOOST_ONNX_MODEL_PATH)
            y = pipe.predict(request.json)
            return jsonify({y.name: y.to_list()})
    except:
        pass
    return make_response("bad request", 400)


@app.route("/metadata/", methods=["GET"])
def metadata():
    try:
        pipe = SalaryPipeline(consts.CATBOOST_ONNX_MODEL_PATH)
        meta = pipe.get_metadata()
        return jsonify(meta)
    except:
        return make_response("bad request", 400)


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
