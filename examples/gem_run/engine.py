"""Deterministic Gem Run course and first-rock race rules; standard library only."""
from __future__ import annotations

import math
import random

SCHEMA_VERSION = 2
LANGUAGES = ("en", "vi")
COLORS = ("blue", "red", "green", "yellow")
WORDS = {"en": dict(blue="blue", red="red", green="green", yellow="yellow"),
         "vi": dict(blue="xanh dương", red="đỏ", green="xanh lá", yellow="vàng")}
ACTIONS = {"en": ("left", "middle", "right"), "vi": ("trái", "giữa", "phải")}


def mission(goal, language, variant=0):
    color = WORDS[language][goal]
    if language == "en":
        texts = [f"Collect {color} gems. Avoid rocks and other colors.",
                 f"Pick up only the {color} gems; avoid rocks.",
                 f"Your target is the {color} gem. Leave the other objects.",
                 f"Choose the lane with a {color} gem. Avoid the other objects."]
    else:
        texts = [f"Nhặt ngọc màu {color}. Tránh đá và ngọc màu khác.",
                 f"Chỉ nhặt ngọc màu {color}; tránh đá.",
                 f"Mục tiêu là viên ngọc màu {color}. Bỏ qua các vật khác.",
                 f"Chọn làn có ngọc màu {color}. Tránh các vật khác."]
    return texts[variant % 4]


def make_row(rng, index, start, end, stage, goal, variant):
    cells = [goal, "rock", rng.choice([c for c in COLORS if c != goal])]
    rng.shuffle(cells)
    order = [0, 1, 2]
    rng.shuffle(order)
    return dict(id=index, start=start, end=end, stage=stage, goal=goal,
                variant=variant, cells=cells, order=order)


def course(seed, stages=9, stage_seconds=4):
    if type(stages) is not int or not 1 <= stages <= 9:
        raise ValueError("stages must be an integer from 1 to 9")
    if type(stage_seconds) is not int or not 1 <= stage_seconds <= 60:
        raise ValueError("stage_seconds must be an integer from 1 to 60")
    rng = random.Random(seed)
    goals = [rng.choice(COLORS) for _ in range(stages * stage_seconds // 2 + 1)]
    for i in range(1, len(goals)):
        if goals[i] == goals[i - 1]:
            goals[i] = COLORS[(COLORS.index(goals[i]) + 1) % len(COLORS)]
    rows = []
    for stage in range(stages):
        rate = 2 ** stage
        for k in range(stage_seconds * rate):
            start = stage * stage_seconds + k / rate
            block = int(start // 2)
            rows.append(make_row(rng, len(rows), start, start + 1 / rate,
                                 stage, goals[block], block % 4))
    return rows


def request_for(row, language):
    actions = ACTIONS[language]
    objects = [("rock" if language == "en" else "đá") if c == "rock"
               else (WORDS[language][c] + " gem" if language == "en"
                     else "ngọc màu " + WORDS[language][c]) for c in row["cells"]]
    if language == "en":
        state = "Gem Run. Next row: " + "; ".join(
            f"{actions[i]} lane: {objects[i]}" for i in range(3)) + "."
        criteria = {actions[i]: f"Move to the {actions[i]} lane." for i in row["order"]}
    else:
        state = "Trò chơi nhặt ngọc. Hàng tiếp theo: " + "; ".join(
            f"làn {actions[i]}: {objects[i]}" for i in range(3)) + "."
        criteria = {actions[i]: f"Chuyển sang làn {actions[i]}." for i in row["order"]}
    return state, {"move": {"type": "choice",
                            "instructions": mission(row["goal"], language, row["variant"]),
                            "criteria": criteria}}


def lane_for(action, language):
    return ACTIONS[language].index(action)


def judge(row, previous_lane, response):
    received = response.get("received") if response else None
    timely = bool(response and response.get("row_id") == row["id"]
                  and isinstance(received, (int, float)) and math.isfinite(received)
                  and row["start"] <= received < row["end"]
                  and type(response.get("lane")) is int and response["lane"] in (0, 1, 2)
                  and not response.get("error"))
    lane = response["lane"] if timely else previous_lane
    cell = row["cells"][lane]
    return dict(row_id=row["id"], lane=lane, previous_lane=previous_lane,
                deadline_miss=not timely, controlled_hit=timely and cell == row["goal"],
                physical_hit=cell == row["goal"], collision=cell == "rock",
                wrong_gem=cell not in ("rock", row["goal"]),
                applied_at=received if timely else None)


class Race:
    """Only the current revealed row can be requested; the first rock is terminal."""

    def __init__(self, rows):
        self.course = rows
        self.results, self.answers, self.issued_rows = [], {}, []
        self.issued_at = {}
        self.index, self.lane, self.busy = 0, 1, None
        self.finished = not rows
        self.stop_reason = "completed" if self.finished else None
        self.stopped_at = 0.0 if self.finished else None

    def tick(self, now, response=None, *, allow_request=True):
        if self.finished:
            return None
        if response is not None and self.busy is not None and response.get("row_id") == self.busy:
            received = response.get("received")
            if isinstance(received, (int, float)) and self.issued_at[self.busy] <= received <= now:
                self.answers[self.busy] = response
                self.busy = None
        while self.index < len(self.course) and self.course[self.index]["end"] <= now:
            row = self.course[self.index]
            outcome = judge(row, self.lane, self.answers.get(row["id"]))
            self.results.append(outcome)
            self.lane = outcome["lane"]
            self.index += 1
            if outcome["collision"] or self.index == len(self.course):
                self.finished = True
                self.stop_reason = "collision" if outcome["collision"] else "completed"
                self.stopped_at = row["end"]
                return None
        row = self.course[self.index]
        if (allow_request and now >= row["start"] and self.busy is None
                and row["id"] not in self.issued_at):
            self.busy = row["id"]
            self.issued_rows.append(row["id"])
            self.issued_at[row["id"]] = now
            return row["id"]
        return None

    def cancel_unsent(self, row_id):
        """Undo reservation when scheduler preemption crossed the send deadline."""
        if self.busy != row_id or self.issued_rows[-1:] != [row_id]:
            raise ValueError("Only the current unsent request can be cancelled")
        self.busy = None
        self.issued_rows.pop()
        del self.issued_at[row_id]


def rules(stages=9, stage_seconds=4):
    return dict(version=2, initial_lane=1, stop_on_first_rock=True,
                stages=stages, stage_seconds=stage_seconds,
                rows_per_second=[2 ** i for i in range(stages)],
                planned_duration=stages * stage_seconds,
                scoring="One point for the target gem with a valid current-row answer received strictly before its deadline; all other rows score zero.",
                execution="One inference in flight. Deadlines never pause. Late or invalid replies retain the previous lane. Skip expired rows. Stop at the first rock; no respawn.",
                inputs="Only the newly revealed row, target instruction and shuffled absolute lane actions; no future rows or recommended action.",
                timing="Monotonic elapsed seconds measured in the parent. Response latency includes IPC, tokenization and inference. Loading and warmup are excluded.")


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def summarize(rows, results, calls):
    def totals(outcomes, responses):
        latencies = [c["response_ms"] for c in responses if not c.get("error")]
        return dict(rows=len(outcomes), timely_hits=sum(r["controlled_hit"] for r in outcomes),
                    deadline_misses=sum(r["deadline_miss"] for r in outcomes),
                    wrong_on_time=sum(not r["deadline_miss"] and not r["controlled_hit"] for r in outcomes),
                    collisions=sum(r["collision"] for r in outcomes),
                    physical_hits=sum(r["physical_hit"] for r in outcomes), calls=len(responses),
                    backend_errors=sum(bool(c.get("error")) for c in responses),
                    p50_ms=percentile(latencies, .5), p95_ms=percentile(latencies, .95))

    summary = totals(results, calls)
    summary["stages"] = []
    for stage in sorted({r["stage"] for r in rows}):
        ids = {r["id"] for r in rows if r["stage"] == stage}
        stage_results = [r for r in results if r["row_id"] in ids]
        stage_calls = [c for c in calls if c["row_id"] in ids]
        summary["stages"].append(dict(stage=stage, rate=2 ** stage, **totals(stage_results, stage_calls)))
    return summary
