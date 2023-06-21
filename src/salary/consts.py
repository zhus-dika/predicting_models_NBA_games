from typing import Final


RAW_DIR: Final = "data/raw/advanced_plus_totals"
RAW_FILENAME_SUFFIX: Final = "_advanced_plus_totals.csv"

INTERIM_DIR: Final = "data/interim/salary"

PREPARED_TRAIN_PATH: Final = f"{INTERIM_DIR}/train.csv"
PREPARED_TEST_PATH: Final = f"{INTERIM_DIR}/test.csv"

PREPROCESSOR_PATH: Final = f"dags/salary/preprocessor.pkl"

PROCESSED_DIR: Final = "data/processed/salary"
PROCESSED_TRAIN_PATH: Final = f"{PROCESSED_DIR}/train_features_target.csv"
PROCESSED_TEST_PATH: Final = f"{PROCESSED_DIR}/test_features_target.csv"

DAGS_DIR: Final = "dags/salary"
MODELS_DIR: Final = "models/salary"

CATBOOST_DAGS_DIR: Final = f"{DAGS_DIR}/catboost"
CATBOOST_MODEL_PATH: Final = f"{MODELS_DIR}/catboost.cbm"
CATBOOST_METRICS_PATH: Final = f"{CATBOOST_DAGS_DIR}/metrics.json"
CATBOOST_ONNX_MODEL_PATH: Final = f"{MODELS_DIR}/catboost.onnx"
