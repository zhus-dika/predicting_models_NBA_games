from typing import Final


MLFLOW_TRACKING_URI = "http://localhost:5000"

RAW_DIR: Final = "data/raw/advanced_plus_totals"
RAW_FILENAME_SUFFIX: Final = "_advanced_plus_totals.csv"
INTERIM_DIR: Final = "data/interim/salary"
PROCESSED_DIR: Final = "data/processed/salary"
DAGS_DIR: Final = "dags/salary"
MODELS_DIR: Final = "models/salary"

CB_PREPARED_TRAIN_PATH: Final = "data/interim/salary/cb_train.csv"
CB_PREPARED_TEST_PATH: Final = "data/interim/salary/cb_test.csv"
CB_PREPROCESSOR_PATH: Final = "dags/salary/catboost/preprocessor.pkl"
CB_PROCESSED_TRAIN_PATH: Final = "data/processed/salary/cb_train_features_target.csv"
CB_PROCESSED_TEST_PATH: Final = "data/processed/salary/cb_test_features_target.csv"
CB_MODEL_PATH: Final = "models/salary/catboost.cbm"
CB_TRAIN_PARAMS_PATH: Final = "models/salary/catboost_params.json"
CB_METRICS_PATH: Final = "dags/salary/catboost/metrics.json"
CB_ONNX_MODEL_PATH: Final = "models/salary/catboost.onnx"

NET_PREPARED_TRAIN_PATH: Final = "data/interim/salary/net_train.csv"
NET_PREPARED_TEST_PATH: Final = "data/interim/salary/net_test.csv"
NET_FEATURES_PREPROCESSOR_PATH: Final = "dags/salary/net/features_preprocessor.pkl"
NET_TARGET_PREPROCESSOR_PATH: Final = "dags/salary/net/target_preprocessor.pkl"
NET_PROCESSED_TRAIN_PATH: Final = "data/processed/salary/net_train_features_target.csv"
NET_PROCESSED_TEST_PATH: Final = "data/processed/salary/net_test_features_target.csv"
NET_MODEL_PATH: Final = "models/salary/net.pt"
NET_TRAIN_PARAMS_PATH: Final = "models/salary/net_params.json"
NET_METRICS_PATH: Final = "dags/salary/net/metrics.json"
