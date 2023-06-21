import click
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from . import consts, utils
from .params import Preprocessing


@click.command()
@click.option("--in-train-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.PREPARED_TRAIN_PATH)
@click.option("--in_test-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.PREPARED_TEST_PATH)
@click.option("--out-train-path", type=click.Path(dir_okay=False), default=consts.PROCESSED_TRAIN_PATH)
@click.option("--out-test-path", type=click.Path(dir_okay=False), default=consts.PROCESSED_TEST_PATH)
@click.option("--out-preprocessor-path", type=click.Path(dir_okay=False), default=consts.PREPROCESSOR_PATH)
@click.option("--target", type=str, required=True)
def featurize(in_train_path: str, in_test_path: str, out_train_path: str, out_test_path: str,
              out_preprocessor_path: str, target: str) -> None:
    in_train = utils.load_data(in_train_path)
    in_test = utils.load_data(in_test_path)

    config = Preprocessing()
    preprocessor, train, test = process(train=in_train, test=in_test, numeric_cols=config.numeric_cols,
                                        cat_cols=config.cat_cols, target=target)

    utils.save_data(train, path=out_train_path)
    utils.save_data(test, path=out_test_path)
    utils.save_preprocessor(preprocessor, path=out_preprocessor_path)


def process(train: DataFrame, test: DataFrame, numeric_cols: list[str], cat_cols: list[str], target: str
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
    featurize()
