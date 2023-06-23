from typing import Any

from pandas import DataFrame, Series
from onnxruntime import InferenceSession
from mlprodict.onnx_conv import get_inputs_from_data

from ..salary.params import Base


class SalaryModel:
    def __init__(self, path: str):
        self.path = path

    def predict(self, x: str | DataFrame) -> Series:
        if x is str:
            x = DataFrame(x, index=[0])

        session = self.create_session()

        cols = [col.name for col in session.get_inputs()]
        features = x[cols]

        inputs = get_inputs_from_data(features)
        output = session.run(None, inputs)[0]

        config = Base()
        y = output.squeeze() if output.size > 1 else output[0]
        return Series(y, name=config.target)

    def get_metadata(self) -> Any:
        session = self.create_session()
        metadata = session.get_modelmeta().custom_metadata_map
        return metadata

    def create_session(self) -> InferenceSession:
        return InferenceSession(self.path)
