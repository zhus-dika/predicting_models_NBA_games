from catboost import CatBoostRegressor
import pytest
import torch

from salary import consts, utils
from salary.params import Net
from salary.net import Regressor


@pytest.fixture(scope="module")
def net_config() -> Net:
    return Net()


def test_catboost_model():
    regressor = CatBoostRegressor().load_model(consts.CB_MODEL_PATH)
    assert regressor.is_fitted()


def test_net_model(net_config: Net):
    params = utils.load_params(consts.NET_TRAIN_PARAMS_PATH)
    num_inputs = params["num_inputs"]
    width = params["width"]
    dropout = params["dropout"]
    assert width == net_config.width and dropout == net_config.dropout

    regressor = Regressor(num_inputs=num_inputs, width=width, dropout=dropout)
    regressor.load_state_dict(torch.load(consts.NET_MODEL_PATH))
    assert regressor.eval() == regressor
