from typing import Optional

import click
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from optuna import create_study
from optuna.integration.mlflow import MLflowCallback
from optuna.samplers import TPESampler
import mlflow
from mlflow.models import infer_signature

from . import consts, utils


@click.group()
def cli():
    pass


@cli.command("catboost")
@click.option("--in-train-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.PROCESSED_TRAIN_PATH)
@click.option("--out-model-path", type=click.Path(dir_okay=False, writable=True), default=consts.CATBOOST_MODEL_PATH)
@click.option("--target", type=str, required=True)
@click.option("--metric-name", type=str, required=True)
@click.option("--valid-size", type=click.FloatRange(min=0.0, max=1.0), required=True)
@click.option("--early-stopping-rounds", type=int, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--n-trials", type=int)
@click.option("--timeout", type=float)
def train_catboost(in_train_path: str, out_model_path: str, target: str, metric_name: str, valid_size: float,
                   early_stopping_rounds: int, seed: int, n_trials: Optional[int] = None,
                   timeout: Optional[float] = None) -> None:
    x, y = utils.load_features_target(in_train_path, target=target)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_size, shuffle=True, random_state=seed)

    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(uri=tracking_uri)
    mlflow_callback = MLflowCallback(tracking_uri=tracking_uri, metric_name=metric_name, mlflow_kwargs={"nested": True},
                                     create_experiment=False)

    @mlflow_callback.track_in_mlflow()
    def objective(trial):
        params = {
            "loss_function": trial.suggest_categorical("loss_function", ["RMSE", "MAE"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e0, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 1e0, log=True),
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

        cbr = CatBoostRegressor(silent=True, random_state=seed, **params)
        cbr.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=early_stopping_rounds)

        y_pred = cbr.predict(x_valid)
        metrics = utils.eval_metrics(y_valid, y_pred)

        mlflow.log_metrics(metrics, step=trial.number)
        mlflow.log_params(params)

        return metrics[metric_name]

    mlflow.set_experiment("train_catboost")

    with mlflow.start_run(run_name=out_model_path):
        study = create_study(sampler=TPESampler(seed=seed), direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=-1, callbacks=[mlflow_callback])

        regressor = CatBoostRegressor(silent=True, random_seed=seed, **study.best_trial.params)
        regressor.fit(x_train, y_train)

        dir_path = utils.get_dir_path(out_model_path)

        utils.make_dir(dir_path)
        regressor.save_model(out_model_path, format=utils.get_extension(out_model_path))

        best_params = study.best_trial.params.copy()
        best_params.update({
            "target": target,
            "metric_name": metric_name,
            "valid_size": valid_size,
            "early_stopping_rounds": early_stopping_rounds,
            "seed": seed
        })
        utils.save_params(f"{dir_path}/params.json", params=best_params)

        pred = regressor.predict(x_valid)

        mlflow.log_params(best_params)
        mlflow.log_metrics(utils.eval_metrics(y_valid, pred))
        mlflow.log_artifact(f"{consts.CATBOOST_DAGS_DIR}/dvc.yaml")

        signature = infer_signature(x_valid, pred)
        mlflow.catboost.log_model(regressor, artifact_path="salary/catboost", signature=signature)


if __name__ == "__main__":
    cli()
