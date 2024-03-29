import click
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from . import consts, utils
from .params import Preprocessing


@click.group()
def cli():
    pass


@cli.command("catboost")
@click.option("--in-train-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.CB_PREPARED_TRAIN_PATH)
@click.option("--in-test-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.CB_PREPARED_TEST_PATH)
@click.option("--out-train-path", type=click.Path(dir_okay=False),
              default=consts.CB_PROCESSED_TRAIN_PATH)
@click.option("--out-test-path", type=click.Path(dir_okay=False),
              default=consts.CB_PROCESSED_TEST_PATH)
@click.option("--out-preprocessor-path", type=click.Path(dir_okay=False),
              default=consts.CB_PREPROCESSOR_PATH)
def preprocess_catboost(in_train_path: str, in_test_path: str, out_train_path: str, out_test_path: str,
                        out_preprocessor_path: str) -> None:
    config = Preprocessing()
    preprocessor, train, test = featurize(in_train_path=in_train_path, in_test_path=in_test_path, config=config)

    utils.save_data(train, path=out_train_path)
    utils.save_data(test, path=out_test_path)
    utils.save_preprocessor(preprocessor, path=out_preprocessor_path)


@cli.command("net")
@click.option("--in-train-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.NET_PREPARED_TRAIN_PATH)
@click.option("--in-test-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.NET_PREPARED_TEST_PATH)
@click.option("--out-train-path", type=click.Path(dir_okay=False),
              default=consts.NET_PROCESSED_TRAIN_PATH)
@click.option("--out-test-path", type=click.Path(dir_okay=False),
              default=consts.NET_PROCESSED_TEST_PATH)
@click.option("--out-features-preprocessor-path", type=click.Path(dir_okay=False),
              default=consts.NET_FEATURES_PREPROCESSOR_PATH)
@click.option("--out-target-preprocessor-path", type=click.Path(dir_okay=False),
              default=consts.NET_TARGET_PREPROCESSOR_PATH)
def preprocess_net(in_train_path: str, in_test_path: str, out_train_path: str, out_test_path: str,
                   out_features_preprocessor_path: str, out_target_preprocessor_path: str) -> None:
    config = Preprocessing()
    features_preprocessor, train, test = featurize(in_train_path=in_train_path, in_test_path=in_test_path,
                                                   config=config)
    target_preprocessor = MinMaxScaler().fit(train[config.target].values.reshape(-1, 1))

    utils.save_data(train, path=out_train_path)
    utils.save_data(test, path=out_test_path)
    utils.save_preprocessor(features_preprocessor, path=out_features_preprocessor_path)
    utils.save_preprocessor(target_preprocessor, path=out_target_preprocessor_path)


def featurize(in_train_path: str, in_test_path: str, config: Preprocessing
              ) -> (ColumnTransformer, DataFrame, DataFrame):
    in_train = utils.load_data(in_train_path)
    in_test = utils.load_data(in_test_path)

    preprocessor, train, test = preprocess(train=in_train, test=in_test, numeric_cols=config.numeric_cols,
                                           cat_cols=config.cat_cols, target=config.target)

    return preprocessor, train, test


def preprocess(train: DataFrame, test: DataFrame, numeric_cols: list[str], cat_cols: list[str], target: str
               ) -> (ColumnTransformer, DataFrame, DataFrame):
    transformers = [
        ("numeric_std", StandardScaler(), numeric_cols),
        ("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float), cat_cols)
    ]

    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough", verbose=False)
    preprocessor.set_output(transform="pandas")

    cols = numeric_cols + cat_cols + [target]
    train_features_target = utils.remove_prefix_remainder(preprocessor.fit_transform(train[cols]))
    test_features_target = utils.remove_prefix_remainder(preprocessor.transform(test[cols]))

    return preprocessor, train_features_target, test_features_target


if __name__ == "__main__":
    cli()
