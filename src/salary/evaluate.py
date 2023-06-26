import click
import torch
from catboost import CatBoostRegressor

from . import consts, utils
from .net import Regressor


@click.group()
def cli():
    pass


@cli.command("catboost")
@click.option("--in-model_path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.CB_MODEL_PATH)
@click.option("--in-test-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.CB_PROCESSED_TEST_PATH)
@click.option("--out-metrics-path", type=click.Path(dir_okay=False, writable=True),
              default=consts.CB_METRICS_PATH)
@click.option("--target", type=str, required=True)
def evaluate_catboost(in_model_path: str, in_test_path: str, out_metrics_path: str, target: str) -> None:
    regressor = CatBoostRegressor().load_model(in_model_path)

    x, y = utils.load_features_target(in_test_path, target)
    pred = regressor.predict(x)

    metrics = utils.eval_metrics(y, pred)
    utils.save_metrics(metrics, path=out_metrics_path)


@cli.command("net")
@click.option("--in-model_path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.NET_MODEL_PATH)
@click.option("--in-test-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.NET_PROCESSED_TEST_PATH)
@click.option("--in-params-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.NET_TRAIN_PARAMS_PATH)
@click.option("--in-target-preprocessor-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.NET_TARGET_PREPROCESSOR_PATH)
@click.option("--out-metrics-path", type=click.Path(dir_okay=False, writable=True),
              default=consts.NET_METRICS_PATH)
@click.option("--target", type=str, required=True)
def evaluate_net(in_model_path: str, in_test_path: str, in_params_path: str, in_target_preprocessor_path: str,
                 out_metrics_path: str, target: str) -> None:
    params = utils.load_params(in_params_path)
    num_inputs = params["num_inputs"]
    width = params["width"]
    dropout = params["dropout"]

    regressor = Regressor(num_inputs=num_inputs, width=width, dropout=dropout)
    regressor.load_state_dict(torch.load(in_model_path))
    regressor.eval()

    x, y = utils.load_features_target(in_test_path, target)
    pred = regressor(torch.tensor(x.values, dtype=torch.float32))

    target_preprocessor = utils.load_preprocessor(in_target_preprocessor_path)
    y_pred = target_preprocessor.inverse_transform(pred.detach().cpu().numpy())

    metrics = utils.eval_metrics(y.values.reshape(-1, 1), y_pred)
    utils.save_metrics(metrics, path=out_metrics_path)


if __name__ == "__main__":
    cli()
