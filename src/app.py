from flask import Flask, request, make_response, jsonify
import onnxruntime as rt
import pandas as pd

from models import salary_model


app = Flask(__name__)
session = rt.InferenceSession('models/salary_model.onnx')


@app.route('/forward/', methods=['POST'])
def forward():
    y = None
    try:
        if request.headers.get('Content-Type') != 'application/json':
            data = request.get_json()
            x = pd.read_json(data)
            y = salary_model.predict(session, x)
    except:
        pass
    return jsonify(y) if y is None else make_response('bad request', 400)


@app.route('/metadata/', methods=['GET'])
def metadata():
    meta = session.get_modelmeta().custom_metadata_map
    return jsonify(meta)


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
