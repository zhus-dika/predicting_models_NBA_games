import click
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from . import consts, utils


@click.group()
def cli():
    pass


@cli.command("catboost")
@click.option("--in-model_path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.CATBOOST_MODEL_PATH)
@click.option("--in-test-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.PROCESSED_TEST_PATH)
@click.option("--out-metrics-path", type=click.Path(dir_okay=False, writable=True),
              default=consts.CATBOOST_METRICS_PATH)
@click.option("--target", type=str, required=True)
def evaluate_catboost(in_model_path: str, in_test_path: str, out_metrics_path: str, target: str) -> None:
    regressor = CatBoostRegressor().load_model(in_model_path)
    assert regressor.is_fitted()

    x, y = utils.load_features_target(in_test_path, target)
    pred = regressor.predict(x)

    r2 = r2_score(y, pred)

    metrics = {"r2": r2}
    utils.save_metrics(out_metrics_path, metrics=metrics)


if __name__ == "__main__":
    cli()
