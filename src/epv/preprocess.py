import numpy
from pandas import DataFrame, concat

from . import consts, params
from ..salary import utils


def preprocess() -> None:
    config = params.EPV()
    data = utils.load_data(consts.EPV_PREPARED_PATH, numeric32=True)

    def flat(sequences: list) -> DataFrame:
        data_frames = []

        for seq in sequences:
            seq_id = seq[0]
            df = seq[1]
            df["game_id"] = seq_id[0]
            df["event_id"] = seq_id[1]
            df["event_type"] = seq_id[2]
            data_frames.append(df)

        return concat(data_frames, axis=0, join="outer", ignore_index=True, copy=False)

    def split(sequences: list, size: float) -> (DataFrame, DataFrame):
        idx = numpy.random.choice(len(sequences), size=int(len(sequences) * size), replace=False)
        first = [sequences[i] for i in range(len(sequences)) if i not in idx]
        second = [sequences[i] for i in idx]
        assert len(first) + len(second) == len(sequences)

        return first, second

    seqs = [(seq_id, moment[config.cols]) for seq_id, moment in data.groupby(["game_id", "event_id", "event_type"])]
    train, test = split(seqs, config.test_size)
    train, valid = split(train, config.valid_size)

    train, valid, test = flat(train), flat(valid), flat(test)
    utils.save_data(train, consts.EPV_PROCESSED_TRAIN_PATH)
    utils.save_data(valid, consts.EPV_PROCESSED_VALID_PATH)
    utils.save_data(test, consts.EPV_PROCESSED_TEST_PATH)


if __name__ == "__main__":
    preprocess()
