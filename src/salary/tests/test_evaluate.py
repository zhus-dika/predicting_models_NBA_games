from salary import consts, utils


def test_catboost_metrics():
    metrics = utils.load_metrics(consts.CATBOOST_METRICS_PATH)
    assert len(metrics) > 0
