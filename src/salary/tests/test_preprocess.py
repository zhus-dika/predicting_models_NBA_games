from typing import Optional

import numpy
import pytest

from salary import consts, utils
from salary.params import Preprocessing


@pytest.fixture(scope="module")
def config() -> Preprocessing:
    return Preprocessing()


@pytest.mark.parametrize("prepared_train_path,prepared_test_path,processed_train_path,processed_test_path", [
    (consts.CB_PREPARED_TRAIN_PATH, consts.CB_PREPARED_TEST_PATH,
     consts.CB_PROCESSED_TRAIN_PATH, consts.CB_PROCESSED_TEST_PATH),
    (consts.NET_PREPARED_TRAIN_PATH, consts.NET_PREPARED_TEST_PATH,
     consts.NET_PROCESSED_TRAIN_PATH, consts.NET_PROCESSED_TEST_PATH)
])
def test_processed_train_test(config: Preprocessing, prepared_train_path: str, prepared_test_path: str,
                              processed_train_path: str, processed_test_path: str):
    train = utils.load_data(processed_train_path)
    test = utils.load_data(processed_test_path)

    assert all(train.columns == test.columns)
    assert len(train.select_dtypes(exclude="number").columns) == 0
    assert len(train.select_dtypes(include="number").columns) > len(config.numeric_cols) + len(config.cat_cols)
    assert config.target in train.columns.values

    assert len(train) == len(utils.load_data(prepared_train_path))
    assert len(test) == len(utils.load_data(prepared_test_path))


@pytest.mark.parametrize("features_preprocessor_path,target_preprocessor_path,prepared_path,processed_path", [
    (consts.CB_PREPROCESSOR_PATH, None, consts.CB_PREPARED_TRAIN_PATH, consts.CB_PROCESSED_TRAIN_PATH),
    (consts.NET_FEATURES_PREPROCESSOR_PATH, consts.NET_TARGET_PREPROCESSOR_PATH,
     consts.NET_PREPARED_TRAIN_PATH, consts.NET_PROCESSED_TRAIN_PATH)
])
def test_preprocessor(config: Preprocessing, features_preprocessor_path: str, target_preprocessor_path: Optional[str],
                      prepared_path: str, processed_path: str):
    preprocessor = utils.load_preprocessor(features_preprocessor_path)
    prepared = utils.load_data(prepared_path)
    processed = utils.load_data(processed_path)
    assert numpy.allclose(utils.remove_prefix_remainder(preprocessor.transform(prepared)), processed)

    if target_preprocessor_path is not None:
        target_preprocessor = utils.load_preprocessor(target_preprocessor_path)
        targets = processed[config.target].values.reshape(-1, 1)
        transformed = target_preprocessor.transform(targets)
        assert numpy.allclose(targets, target_preprocessor.inverse_transform(transformed))
