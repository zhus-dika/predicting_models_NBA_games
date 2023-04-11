from flask import Flask, request, make_response, jsonify
import pandas as pd
from models.salary_model import SalaryModel

SALARY_MODEL_FILEPATH = 'models/salary_model.onnx'

app = Flask(__name__)


@app.route('/forward/', methods=['POST'])
def forward():
    try:
        if request.headers.get('Content-Type') == 'application/json':
            x = pd.DataFrame(request.get_json(), index=[0])
            y = SalaryModel(SALARY_MODEL_FILEPATH).predict(x)
            return jsonify({y.name: y.to_list()})
    except:
        pass
    return make_response('bad request', 400)


@app.route('/metadata/', methods=['GET'])
def metadata():
    try:
        meta = SalaryModel(SALARY_MODEL_FILEPATH).get_metadata()
        return jsonify(meta)
    except:
        return make_response('bad request', 400)


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
