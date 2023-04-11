import os
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor
from catboost.utils import convert_to_onnx_object

from onnx.helper import get_attribute_value
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
from mlprodict.onnx_conv import guess_schema_from_data, get_inputs_from_data
import onnxruntime as rt

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)


# http://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_catboost.html
def skl2onnx_convert_catboost(scope, operator, container):
    """
    CatBoost returns an ONNX graph with a single node.
    This function adds it to the main graph.
    """
    onx = convert_to_onnx_object(operator.raw_operator)
    opsets = {d.domain: d.version for d in onx.opset_import}
    if '' in opsets and opsets[''] >= container.target_opset:
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
        op_domain=node.domain, op_version=opsets.get(node.domain, None),
        **atts)


class SalaryModel(object):
    def __init__(self, filepath: str):
        self.best = None
        self.filepath = filepath

    @staticmethod
    def load_dataset(path: str) -> (pd.DataFrame, pd.DataFrame, pd.Series):
        dfs = []
        for year in range(2011, 2022):
            filepath = os.path.join(path, f'{year}_advanced_plus_totals.csv')
            df = pd.read_csv(filepath)
            df['year'] = year
            df.drop('Salary', axis=1, inplace=True)
            dfs.append(df)
        data = pd.concat(dfs, axis=0, join='outer', ignore_index=True)
        return data.drop('SalaryAdj', axis=1), data['SalaryAdj']

    @staticmethod
    def fix_column_name(name: str) -> str:
        name = name.strip()
        if name and name[0].isdigit():
            name = '_' + name
        return name.replace('%', 'percent').replace('/', '_')

    @staticmethod
    def fix_data_column_names(data: pd.DataFrame) -> pd.DataFrame:
        data.columns = map(SalaryModel.fix_column_name, data.columns.values)
        return data

    def fit(self, x: pd.DataFrame, y: pd.Series, cat_features: list, n_trials: int | None = None,
            timeout: float | None = None, seed: Any = None) -> float:
        num_features = x[:1].select_dtypes(include=np.number).columns.values
        num_features = list(filter(lambda f: f not in cat_features, num_features))

        assert not x[cat_features + num_features].isnull().any(axis=None)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)
        x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.4, random_state=seed)

        transformers = [
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
            ('num', StandardScaler(), num_features)
        ]
        early_stopping_rounds = 100

        def objective(trial):
            params = {
                "loss_function": trial.suggest_categorical("loss_function", ["RMSE", "MAE"]),
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e0),
                "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 1e0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                "depth": trial.suggest_int("depth", 1, 10),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
                "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
            }

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            pipe = Pipeline(steps=[
                ('preprocessor', ColumnTransformer(transformers=transformers, remainder='drop')),
                ('regressor', CatBoostRegressor(silent=True, random_seed=seed, **params))
            ])

            pipe_eval_set = [(pipe.steps[0][1].fit(x_train).transform(x_valid), y_valid)]
            pipe.fit(x_train, y_train, regressor__eval_set=pipe_eval_set,
                     regressor__early_stopping_rounds=early_stopping_rounds)

            return r2_score(y_test, pipe.predict(x_test))

        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=-1)

        self.best = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(transformers=transformers, remainder='drop')),
            ('regressor', CatBoostRegressor(silent=True, random_seed=seed, **study.best_trial.params))
        ])

        eval_set = [(self.best.steps[0][1].fit(x_train).transform(x_valid), y_valid)]
        self.best.fit(x_train, y_train, regressor__eval_set=eval_set,
                      regressor__early_stopping_rounds=early_stopping_rounds)

        return r2_score(y_test, self.best.predict(x_test))

    @staticmethod
    def convert_data_for_onnx(data: pd.DataFrame) -> pd.DataFrame:
        x = SalaryModel.fix_data_column_names(data.copy())
        cols = x[:1].select_dtypes(include=np.number).columns.values
        return x.astype(dict.fromkeys(cols, np.float32))

    def save_to_onnx(self, x: pd.DataFrame, score: float):
        update_registered_converter(CatBoostRegressor, alias='CatBoostCatBoostRegressor',
                                    shape_fct=calculate_linear_regressor_output_shapes,
                                    convert_fct=skl2onnx_convert_catboost)

        initial_types = guess_schema_from_data(SalaryModel.convert_data_for_onnx(x))
        onx = convert_sklearn(self.best, initial_types=initial_types)

        meta_timestamp = onx.metadata_props.add()
        meta_timestamp.key, meta_timestamp.value = 'date', str(datetime.now())

        meta_name = onx.metadata_props.add()
        meta_name.key, meta_name.value = 'name', os.path.basename(self.filepath)

        meta_score = onx.metadata_props.add()
        meta_score.key, meta_score.value = 'score', str(round(score, 3))

        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        with open(self.filepath, 'wb') as f:
            f.write(onx.SerializeToString())

        return onx

    def train(self, x: pd.DataFrame, y: pd.Series, n_trials: int | None = None, timeout: float | None = None,
              seed: Any = None) -> float:
        random_state = np.random.get_state()
        np.random.seed(seed)

        cat_cols = ['Pos']
        num_cols = ['year', 'Age', 'G', 'GS', 'MP', 'FG%', '3P', '3P%', '2P%', 'eFG%', 'FT%', 'ORB', 'DRB', 'AST',
                    'STL', 'BLK', 'TOV', 'PF', 'PTS', '3PAr', 'AST%', 'BLK%', 'BPM', 'DBPM', 'DRB%', 'DWS', 'FTr',
                    'ORB%', 'OWS', 'PER', 'STL%', 'TOV%', 'USG%', 'VORP', 'WS/48']

        x = SalaryModel.fix_data_column_names(x[cat_cols + num_cols])
        cat_features = list(map(SalaryModel.fix_column_name, cat_cols))
        score = self.fit(x, y, cat_features=cat_features, n_trials=n_trials, timeout=timeout, seed=seed)

        self.save_to_onnx(x=x[:1], score=score)

        np.random.set_state(random_state)

        return score

    def predict(self, x: pd.DataFrame) -> pd.Series:
        session = rt.InferenceSession(self.filepath)

        data = SalaryModel.convert_data_for_onnx(x)
        data = data[[arg.name for arg in session.get_inputs()]]

        inputs = get_inputs_from_data(data)
        output = session.run(None, inputs)[0]
        return pd.Series(output.squeeze() if output.size > 1 else output[0], name='SalaryAdj')

    def get_metadata(self):
        return rt.InferenceSession(self.filepath).get_modelmeta().custom_metadata_map
