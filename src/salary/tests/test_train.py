from catboost import CatBoostRegressor

from salary import consts


def test_catboost_model():
    regressor = CatBoostRegressor().load_model(consts.CATBOOST_MODEL_PATH)
    assert regressor.is_fitted()
