from dataclasses import dataclass
from typing import Final


RAW_DIR: Final = "data/raw/movement"
EVENTS_DIR: Final = f"{RAW_DIR}/events"

EPV_PREPARED_PATH: Final = "data/interim/epv/prepared.csv"
EPV_PROCESSED_TRAIN_PATH: Final = "data/processed/epv/train.csv"
EPV_PROCESSED_TEST_PATH: Final = "data/processed/epv/test.csv"
EPV_PROCESSED_VALID_PATH: Final = "data/processed/epv/valid.csv"


@dataclass(frozen=True)
class EventType:
    FIELD_GOAL_MADE = 1  # забит бросок с игры (результативная атака)
    FIELD_GOAL_MISS = 2  # промах после броска с игры (неудачная атака с броском)
    FREE_THROW_ATTEMPT = 3  # штрафные броски
    REBOUND = 4  # подбор
    TURNOVER = 5  # аут, потеря или перехват мяча (неудачная атака с потерей мяча)
    FOUL = 6  # фол на атакующем игроке
    VIOLATION = 7  # нарушение атакующей команды (неудачная атака из-за нарушения)
    SUBSTITUTION = 8  # замена
    TIMEOUT = 9  # тайм-аут
    JUMP_BALL = 10  # спорный мяч
    EJECTION = 11  # удаление игрока до конца игры
    PERIOD_BEGIN = 12  # начало периода
    PERIOD_END = 13  # окончание периода
