import os
from typing import Any, Optional
from pathlib import Path
import json

from pandas import DataFrame, Series, read_csv
import joblib
from catboost.utils import convert_to_onnx_object
from onnx import ModelProto
from onnx.helper import get_attribute_value
from sklearn.metrics import r2_score, mean_absolute_error


def fix_column_name(name: str) -> str:
    name = name.strip()

    if name and name[0].isdigit():
        name = "_" + name

    name = name.replace("%", "pct").replace("/", "_")
    return name


def fix_column_names(columns: list[str]) -> list[str]:
    return list(map(fix_column_name, columns))


def remove_prefix_remainder(data: DataFrame) -> DataFrame:
    columns = {name: name.removeprefix("remainder__") for name in data.columns.values if name.startswith("remainder__")}
    data.rename(columns=columns, inplace=True)
    return data


def save_data(data: DataFrame, path: str, index: bool = False) -> None:
    make_dir(os.path.dirname(path))
    data.to_csv(path, index=index)


def load_data(path: str) -> DataFrame:
    return read_csv(path)


def load_features_target(path: str, target: str) -> (DataFrame, Series):
    data = read_csv(path)
    y = data[target]
    data.drop(target, axis=1, inplace=True)
    return data, y


def save_metrics(metrics: dict[str, float], path: str) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f)


def load_metrics(path: str) -> dict[str, float]:
    with open(path, "rb") as f:
        return json.load(f)


def eval_metrics(y_true: Any, y_pred: Any, names: Optional[list[str]] = None) -> dict[str, float]:
    if names is None:
        names = ["r2", "mae"]

    metrics = {
        "r2": r2_score,
        "mae": mean_absolute_error
    }
    return {name: metrics[name](y_true, y_pred) for name in names}


def save_params(params: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(params, f)


def load_params(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def make_dir(dir_path: str) -> Path:
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_extension(path: str) -> str:
    return Path(path).stem


def get_dir_path(path: str) -> str:
    return os.path.dirname(path)


def save_preprocessor(preprocessor: Any, path: str) -> None:
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str) -> Any:
    return joblib.load(path)


def remove_if_exists(path: str) -> None:
    Path(path).unlink(missing_ok=True)


def save_onnx(onx: ModelProto, path: str) -> None:
    remove_if_exists(path)
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())


# http://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_catboost.html
def skl2onnx_convert_catboost(scope, operator, container):
    """
    CatBoost returns an ONNX graph with a single node.
    This function adds it to the main graph.
    """
    onx = convert_to_onnx_object(operator.raw_operator)
    op_sets = {d.domain: d.version for d in onx.opset_import}
    if '' in op_sets and op_sets[''] >= container.target_opset:
        raise RuntimeError(
            "CatBoost uses an opset more recent than the target one.")
    if len(onx.graph.initializer) > 0 or len(onx.graph.sparse_initializer) > 0:
        raise NotImplementedError(
            "CatBoost returns a model initializers. This option is not implemented yet.")
    if (len(onx.graph.node) not in (1, 2) or not onx.graph.node[0].op_type.startswith("TreeEnsemble") or
            (len(onx.graph.node) == 2 and onx.graph.node[1].op_type != "ZipMap")):
        types = ", ".join(map(lambda n: n.op_type, onx.graph.node))
        raise NotImplementedError(
            f"CatBoost returns {len(onx.graph.node)} != 1 (types={types}). "
            f"This option is not implemented yet.")
    node = onx.graph.node[0]
    atts = {}
    for att in node.attribute:
        atts[att.name] = get_attribute_value(att)
    container.add_node(
        node.op_type, [operator.inputs[0].full_name],
        [operator.outputs[0].full_name],
        op_domain=node.domain, op_version=op_sets.get(node.domain, None),
        **atts)
