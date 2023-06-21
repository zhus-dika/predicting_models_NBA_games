import os

import click
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from . import consts, utils


@click.group()
def cli():
    pass


@cli.command("catboost")
@click.option("--in-train-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.PROCESSED_TRAIN_PATH)
@click.option("--out-model-path", type=click.Path(dir_okay=False, writable=True), default=consts.CATBOOST_MODEL_PATH)
@click.option("--target", type=str, required=True)
@click.option("--valid-size", type=click.FloatRange(min=0.0, max=1.0), required=True)
@click.option("--early-stopping-rounds", type=int, required=True)
@click.option("--seed", type=int, required=True)
def train_catboost(in_train_path: str, out_model_path: str, target: str, valid_size: float, early_stopping_rounds: int,
                   seed: int) -> None:
    x, y = utils.load_features_target(in_train_path, target=target)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_size, shuffle=True, random_state=seed)

    # TODO: optuna, tracking and logging
    regressor = CatBoostRegressor(silent=True, random_seed=seed)
    regressor.fit(x_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_set=[(x_valid, y_valid)])

    utils.make_dir(os.path.dirname(out_model_path))
    regressor.save_model(out_model_path, format=utils.get_extension(out_model_path))


if __name__ == "__main__":
    cli()
