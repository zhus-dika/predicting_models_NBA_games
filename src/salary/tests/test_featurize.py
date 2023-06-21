import numpy
import pytest

from salary import consts, utils
from salary.params import Preprocessing


@pytest.fixture(scope="module")
def config() -> Preprocessing:
    return Preprocessing()


def test_processed_train_test(config: Preprocessing):
    train = utils.load_data(consts.PROCESSED_TRAIN_PATH)
    test = utils.load_data(consts.PROCESSED_TEST_PATH)

    assert all(train.columns == test.columns)
    assert len(train.select_dtypes(exclude="number").columns) == 0
    assert len(train.select_dtypes(include="number").columns) > len(config.numeric_cols) + len(config.cat_cols)
    assert config.target in train.columns.values

    assert len(train) == len(utils.load_data(consts.PREPARED_TRAIN_PATH))
    assert len(test) == len(utils.load_data(consts.PREPARED_TEST_PATH))


def test_preprocessor():
    preprocessor = utils.load_preprocessor(consts.PREPROCESSOR_PATH)
    in_test = utils.load_data(consts.PREPARED_TRAIN_PATH)
    out_test = utils.load_data(consts.PROCESSED_TRAIN_PATH)
    assert numpy.allclose(utils.remove_prefix_remainder(preprocessor.transform(in_test)), out_test)
