import pytest

from salary import consts, params, utils


@pytest.mark.parametrize("metrics_path,metric_name", [
    (consts.CB_METRICS_PATH, params.Catboost().metric_name),
    (consts.NET_METRICS_PATH, params.Net().metric_name)
])
def test_metrics(metrics_path: str, metric_name: str):
    metrics = utils.load_metrics(metrics_path)
    assert len(metrics) > 0 and metric_name in metrics.keys()
