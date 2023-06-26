from dataclasses import dataclass, field
from typing import Optional

from . import utils


@dataclass(frozen=True)
class Base:
    seed: int = 42
    target: str = "SalaryAdj"


@dataclass(frozen=True)
class Preprocessing(Base):
    test_size: float = 0.3
    numeric_cols: list[str] = field(default_factory=lambda: utils.fix_column_names([
        "year", "Age", "G", "GS", "MP", "FG%", "3P", "3P%", "2P%", "eFG%", "FT%", "ORB", "DRB", "AST", "STL", "BLK",
        "TOV", "PF", "PTS", "3PAr", "AST%", "BLK%", "BPM", "DBPM", "DRB%", "DWS", "FTr", "ORB%", "OWS", "PER", "STL%",
        "TOV%", "USG%", "VORP", "WS/48"
    ]))
    cat_cols: list[str] = field(default_factory=lambda: utils.fix_column_names([
        "Pos"
    ]))


@dataclass(frozen=True)
class Catboost(Preprocessing):
    metric_name: str = "r2"
    valid_size: float = 0.25
    early_stopping_rounds: int = 100
    n_trials: Optional[int] = 10
    timeout: Optional[float] = 1000


@dataclass(frozen=True)
class Net(Base):
    metric_name: str = "r2"
    valid_size: float = 0.25
    width: int = 64
    dropout: float = 0.3
    n_epochs: int = 100
    batch_size: int = 128
