import pandas.api.types
import pytest

from salary import consts, utils
from salary.params import Preprocessing


@pytest.fixture(scope="module")
def config() -> Preprocessing:
    return Preprocessing()


def test_prepared_train_test(config: Preprocessing):
    train = utils.load_data(consts.PREPARED_TRAIN_PATH)
    test = utils.load_data(consts.PREPARED_TEST_PATH)

    assert all(train.columns == test.columns)

    numeric_cols, cat_cols, target = config.numeric_cols, config.cat_cols, config.target

    assert all([pandas.api.types.is_numeric_dtype(dt) for dt in train[numeric_cols].dtypes.values])
    assert target not in numeric_cols

    assert all([pandas.api.types.is_object_dtype(dt) for dt in train[cat_cols].dtypes.values])
    assert target not in cat_cols

    assert pandas.api.types.is_numeric_dtype(train[target])

    assert round(len(test) / (len(train) + len(test)), 3) == config.test_size
