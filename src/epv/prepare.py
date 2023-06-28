from pathlib import Path
import json

from pandas import DataFrame, read_csv, concat
import py7zlib

from . import consts
from .consts import EventType
from ..salary import utils


def prepare() -> None:
    dataset = load_dataset(consts.RAW_DIR, consts.EVENTS_DIR, max_games_count=100)
    utils.save_data(dataset, path=consts.EPV_PREPARED_PATH)


def load_dataset(games_dir_path: str, events_dir_path: str, max_games_count: int) -> DataFrame:
    data_frames = []

    game_paths = list(map(str, Path(games_dir_path).glob("*.7z")))
    for count, game_path in enumerate(game_paths, start=1):
        if count > max_games_count:
            break

        game, filename = extract_game_data(game_path)

        if game is None:
            print(f"skip: {game_path}")
            continue

        event_path = Path(events_dir_path).joinpath(filename).with_suffix(".csv")
        events = read_csv(event_path)

        print(f"{count} of {len(game_paths)}: {game_path}, {event_path}")

        game_moments = get_game_moments(game, events)
        offensive_moments = get_offensive_moments(game_moments)

        data_frames.append(offensive_moments)

    dataset = concat(data_frames, axis=0, join="outer", ignore_index=True, copy=False)
    return dataset


def extract_game_data(path: str) -> (dict, str):
    with open(path, "rb") as f:
        archive = py7zlib.Archive7z(f)
        names = archive.getnames()

        # FIXME: assert len(names) == 1
        if len(names) != 1:
            return None, None

        filename = names[0]
        data = archive.getmember(filename).read().decode()
        return json.loads(data), filename


def get_game_moments(game: dict, events: DataFrame) -> DataFrame:
    game_moments = []

    game_id = game["gameid"]
    for event in game["events"]:
        event_id = int(event["eventId"])

        event_ext_mask = events["EVENTNUM"] == event_id
        assert event_ext_mask.sum() in [0, 1]  # doesn't exist or is unique

        event_ext = events[event_ext_mask]
        if event_ext.empty:
            continue

        # player_to_team = {p["playerid"]: event["visitor"]["abbreviation"] for p in event["visitor"]["players"]}
        # player_to_team.update({p["playerid"]: event["home"]["abbreviation"] for p in event["home"]["players"]})

        for moment in event["moments"]:
            assert moment[4] is None

            ball = None  # ball may be out of bounds
            if moment[5][0][0] == -1:
                ball = moment[5][0]
                assert ball[0] == -1 and ball[1] == -1

            players = moment[5][(0 if ball is None else 1):]
            assert sum([p[4] for p in players]) == 0  # player's z-coordinate is always zero

            if len(players) < 10:
                continue
            assert len(players) == 10

            game_moment = {
                "game_id": game_id,
                "event_id": event_id,
                "event_type": event_ext["EVENTMSGTYPE"].values[0],
                # "quarter": moment[0],
                # "timestamp": moment[1],
                # "game_clock": moment[2],
                "shot_clock": moment[3]
            }

            for i, player in enumerate(players, start=1):
                game_moment.update({
                    # f"p{i}_id": player[1],
                    f"p{i}_x": player[2],
                    f"p{i}_y": player[3],
                    # f"p{i}_team": player_to_team[player[1]]
                })

            if ball is None:
                bx, by, bz = -1, -1, -1
            else:
                bx, by, bz = ball[2], ball[3], ball[4]
            game_moment.update({"b_x": bx, "b_y": by, "b_z": bz})

            game_moments.append(game_moment)

    df_game_moments = DataFrame.from_records(game_moments)
    assert len(df_game_moments) > 0

    return df_game_moments


def get_offensive_moments(moments: DataFrame) -> DataFrame:
    offensive_moments = []

    for (game_id, event_id, event_type), event_moments in moments.groupby(["game_id", "event_id", "event_type"]):
        if event_type not in [EventType.FIELD_GOAL_MADE, EventType.FIELD_GOAL_MISS, EventType.TURNOVER,
                              EventType.FOUL, EventType.VIOLATION]:
            continue

        min_shot_clock = 24.0
        for _, moment in event_moments[event_moments["shot_clock"].notnull()].iterrows():
            shot_clock = moment["shot_clock"]

            if shot_clock > min_shot_clock and shot_clock > 23:
                break

            min_shot_clock = shot_clock

            offensive_moments.append(moment)

    # TODO: check duplicates
    df_offensive_moments = DataFrame(offensive_moments)
    if len(df_offensive_moments) > 0:
        df_offensive_moments = df_offensive_moments.astype({"event_id": int, "event_type": int}, copy=False)
    assert not df_offensive_moments.isnull().any().any()

    return df_offensive_moments


if __name__ == "__main__":
    prepare()
