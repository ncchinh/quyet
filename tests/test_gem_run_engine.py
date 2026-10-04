import importlib

import pytest


def engine():
    try:
        return importlib.import_module("examples.gem_run.engine")
    except ModuleNotFoundError:
        pytest.fail("The portable Gem Run engine is not implemented")


def row(index, cells, *, start=None, end=None):
    return dict(id=index, start=float(index if start is None else start),
                end=float(index + 1 if end is None else end), stage=0,
                goal="blue", variant=0, cells=cells, order=[2, 0, 1])


def test_first_rock_ends_race_before_next_request_and_preserves_prefix():
    race = engine().Race([
        row(0, ["blue", "red", "rock"]),
        row(1, ["blue", "rock", "red"]),
        row(2, ["rock", "blue", "red"]),
    ])
    assert race.tick(0) == 0
    assert race.tick(.2, dict(row_id=0, received=.2, lane=0)) is None
    assert race.tick(1) == 1
    race.tick(1.2, dict(row_id=1, received=1.2, lane=1))
    assert race.tick(2) is None
    assert race.finished and race.stop_reason == "collision"
    assert race.stopped_at == 2
    assert race.issued_rows == [0, 1]
    assert [r["controlled_hit"] for r in race.results] == [True, False]
    assert race.results[-1]["collision"] is True
    assert race.tick(3, dict(row_id=1, received=2.5, lane=0)) is None
    assert len(race.results) == 2


def test_busy_worker_skips_expired_rows_and_late_answer_cannot_steer():
    race = engine().Race([row(i, ["blue", "red", "rock"]) for i in range(4)])
    assert race.tick(0) == 0
    assert race.tick(2.2) is None
    assert race.tick(2.3, dict(row_id=0, received=2.3, lane=2)) == 2
    assert race.issued_rows == [0, 2]
    assert [r["lane"] for r in race.results] == [1, 1]
    assert all(r["deadline_miss"] for r in race.results)


def test_rock_during_hung_request_stops_at_first_expired_row():
    race = engine().Race([row(i, ["blue", "rock", "red"]) for i in range(4)])
    race.tick(0)
    assert race.tick(4) is None
    assert race.finished and race.stopped_at == 1
    assert len(race.results) == 1
    assert race.issued_rows == [0]


@pytest.mark.parametrize("response", [
    dict(row_id=0, received=1, lane=0),
    dict(row_id=0, received=-.1, lane=0),
    dict(row_id=1, received=.2, lane=0),
    dict(row_id=0, received=.2, lane=3),
    dict(row_id=0, received=.2, lane=True),
    dict(row_id=0, received=.2, lane=0, error="invalid probabilities"),
    None,
])
def test_invalid_or_late_reply_keeps_previous_lane_without_credit(response):
    result = engine().judge(row(0, ["blue", "red", "rock"]), 1, response)
    assert result == dict(row_id=0, lane=1, previous_lane=1, deadline_miss=True,
                          controlled_hit=False, physical_hit=False, collision=False,
                          wrong_gem=True, applied_at=None)


def test_lucky_timeout_gem_is_physical_hit_only():
    result = engine().judge(row(0, ["red", "blue", "rock"]), 1, None)
    assert result["physical_hit"] is True
    assert result["controlled_hit"] is False
    assert result["deadline_miss"] is True


def test_unsolicited_or_future_reply_cannot_control_race():
    race = engine().Race([row(0, ["blue", "red", "rock"])])
    assert race.tick(0, dict(row_id=0, received=.1, lane=0)) == 0
    race.tick(1)
    assert race.results[0]["lane"] == 1


def test_reply_cannot_free_worker_for_a_different_request():
    race = engine().Race([row(i, ["blue", "red", "rock"]) for i in range(3)])
    race.tick(0)
    assert race.tick(1.2, dict(row_id=1, received=1.2, lane=0)) is None
    assert race.busy == 0
    assert race.issued_rows == [0]


def test_invalid_reply_does_not_allow_a_second_request_for_same_row():
    race = engine().Race([row(0, ["blue", "red", "rock"])])
    race.tick(0)
    assert race.tick(.1, dict(row_id=0, received=.1, lane=None, error="invalid")) is None
    assert race.tick(.2) is None
    race.tick(1)
    assert race.issued_rows == [0]
    assert race.results[0]["deadline_miss"] is True


def test_english_and_vietnamese_reveal_same_objects_and_shuffled_actions():
    e = engine()
    r = row(0, ["blue", "rock", "yellow"])
    en_state, en_questions = e.request_for(r, "en")
    vi_state, vi_questions = e.request_for(r, "vi")
    assert "left lane: blue gem; middle lane: rock; right lane: yellow gem" in en_state
    assert "làn trái: ngọc màu xanh dương; làn giữa: đá; làn phải: ngọc màu vàng" in vi_state
    assert list(en_questions["move"]["criteria"]) == ["right", "left", "middle"]
    assert list(vi_questions["move"]["criteria"]) == ["phải", "trái", "giữa"]
    assert "blue" in en_questions["move"]["instructions"]
    assert "xanh dương" in vi_questions["move"]["instructions"]


def test_course_is_seeded_and_uses_stage_deadlines():
    e = engine()
    rows = e.course(701, stages=2, stage_seconds=2)
    assert rows == e.course(701, stages=2, stage_seconds=2)
    assert rows != e.course(702, stages=2, stage_seconds=2)
    assert [(r["start"], r["end"]) for r in rows] == [
        (0, 1), (1, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4)]
    assert all(r["cells"].count(r["goal"]) == 1 and r["cells"].count("rock") == 1 for r in rows)


@pytest.mark.parametrize("stages,seconds", [(0, 4), (10, 4), (1, 0), (1, -2)])
def test_bad_course_limits_fail_before_allocation(stages, seconds):
    with pytest.raises(ValueError):
        engine().course(701, stages=stages, stage_seconds=seconds)
