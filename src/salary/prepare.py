import os
from typing import Any
from pathlib import Path

import click
from pandas import DataFrame, read_csv, concat
from sklearn.model_selection import train_test_split

from . import consts, utils


@click.command()
@click.option("--in-dir", type=click.Path(exists=True, file_okay=False, readable=True), default=consts.RAW_DIR)
@click.option("--in-filename-suffix", type=str, default=consts.RAW_FILENAME_SUFFIX)
@click.option("--out-train-path", type=click.Path(dir_okay=False), default=consts.PREPARED_TRAIN_PATH)
@click.option("--out-test-path", type=click.Path(dir_okay=False), default=consts.PREPARED_TEST_PATH)
@click.option("--test-size", type=click.FloatRange(min=0.0, max=1.0), required=True)
@click.option("--seed", type=int, required=True)
def prepare(in_dir: str, in_filename_suffix: str, out_train_path: str, out_test_path: str, test_size: float,
            seed: Any = None) -> None:
    # TODO: get raw data from dvc remote

    dataset = load_dataset(in_dir, in_filename_suffix)
    dataset.columns = utils.fix_column_names(dataset.columns.values)

    train, test = train_test_split(dataset, test_size=test_size, random_state=seed, shuffle=True)

    utils.save_data(train, path=out_train_path)
    utils.save_data(test, path=out_test_path)


def load_dataset(dir_path: str, filename_suffix: str) -> DataFrame:
    data_frames = []
    for path in Path(dir_path).glob("*" + filename_suffix):
        df = read_csv(path)
        df.drop("Salary", axis=1, inplace=True)

        filename = os.path.basename(path)
        df["year"] = int(filename.removesuffix(filename_suffix))

        data_frames.append(df)

    dataset = concat(data_frames, axis=0, join="outer", ignore_index=True)
    return dataset


if __name__ == "__main__":
    prepare()
