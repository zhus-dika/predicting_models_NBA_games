import os
from datetime import datetime

import click
from catboost import CatBoostRegressor
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
from sklearn.pipeline import Pipeline
from mlprodict.onnx_conv import guess_schema_from_data
from numpy import float32

from . import consts, utils


@click.group()
def cli():
    pass


@cli.command("catboost")
@click.option("--in-preprocessor-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.PREPROCESSOR_PATH)
@click.option("--in-model-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.CATBOOST_MODEL_PATH)
@click.option("--in-data-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.PREPARED_TEST_PATH)
@click.option("--out-onnx-path", type=click.Path(dir_okay=False), default=consts.CATBOOST_ONNX_MODEL_PATH)
@click.option("--target", type=str, required=True)
def catboost_to_onnx(in_preprocessor_path: str, in_model_path: str, in_data_path: str, out_onnx_path: str,
                     target: str) -> None:
    preprocessor = utils.load_preprocessor(in_preprocessor_path)
    model = CatBoostRegressor().load_model(in_model_path)
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    features, _ = utils.load_features_target(in_data_path, target)
    numeric_cols = features.select_dtypes(include="number").columns.values
    features = features.astype(dict.fromkeys(numeric_cols, float32), copy=False, errors="raise")
    initial_types = guess_schema_from_data(features)

    update_registered_converter(CatBoostRegressor, alias="CatBoostCatBoostRegressor",
                                shape_fct=calculate_linear_regressor_output_shapes,
                                convert_fct=utils.skl2onnx_convert_catboost)

    onx = convert_sklearn(pipe, initial_types=initial_types)

    meta_timestamp = onx.metadata_props.add()
    meta_timestamp.key, meta_timestamp.value = "date", str(datetime.now())

    meta_name = onx.metadata_props.add()
    meta_name.key, meta_name.value = "name", os.path.basename(out_onnx_path)

    #meta_score = onx.metadata_props.add()
    #meta_score.key, meta_score.value = "score", str(round(score, 3))

    utils.remove_if_exists(out_onnx_path)
    with open(out_onnx_path, "wb") as f:
        f.write(onx.SerializeToString())


if __name__ == "__main__":
    cli()
