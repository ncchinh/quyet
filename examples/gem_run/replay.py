"""Build the offline Gem Run viewer with only the Python standard library."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import webbrowser


HERE = Path(__file__).resolve().parent
RACE_FILE = re.compile(r"^(.+)_race_(en|vi)_(-?\d+)\.json$")
MODEL_ORDER = ("small_en", "laya_english", "small", "laya_multilingual")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile(course: list[dict]) -> dict:
    if not course:
        raise ValueError("A race course must contain at least one row")
    stages: dict[int, list[dict]] = {}
    previous_end = 0.0
    for index, row in enumerate(course):
        start, end = row["start"], row["end"]
        if (row["id"] != index or not math.isfinite(start) or not math.isfinite(end)
                or start != previous_end or end <= start):
            raise ValueError("Invalid course row order or timing")
        cells = row["cells"]
        if (len(cells) != 3 or cells.count("rock") != 1 or row["goal"] not in cells
                or row["goal"] not in ("blue", "green", "red", "yellow")
                or sorted(row["order"]) != [0, 1, 2]):
            raise ValueError("Invalid course objects or action order")
        stages.setdefault(row["stage"], []).append(row)
        previous_end = end
    if sorted(stages) != list(range(len(stages))):
        raise ValueError("Course stages must be consecutive, starting at zero")
    durations = [rows[-1]["end"] - rows[0]["start"] for rows in stages.values()]
    if any(not math.isclose(d, durations[0]) for d in durations):
        raise ValueError("Course stages must use the same duration")
    rates = [round(1 / (rows[0]["end"] - rows[0]["start"])) for rows in stages.values()]
    return {"stage_seconds": durations[0], "duration_seconds": previous_end,
            "rows_per_second": rates}


def _validate_run(record: dict, course: list[dict]) -> None:
    results = record["results"]
    if not results or len(results) > len(course):
        raise ValueError("A recording must contain its measured result prefix")
    for index, result in enumerate(results):
        if result["row_id"] != index or result["lane"] not in (0, 1, 2):
            raise ValueError("Invalid recorded result order or lane")
        if result["collision"] and index != len(results) - 1:
            raise ValueError("A survival recording must stop at its first rock collision")
    if record["duration"] != course[len(results) - 1]["end"]:
        raise ValueError("Recording duration must end at its final measured row")
    for call in record.get("calls", []):
        if (not 0 <= call["row_id"] < len(course)
                or not math.isfinite(call["issued"]) or not math.isfinite(call["received"])
                or call["received"] < call["issued"]):
            raise ValueError("Invalid recorded response timing")


def load_records(records: Path | None) -> dict:
    """Load the bundled measurements, or compare compatible live-run JSON files."""
    if records is None:
        path = HERE / "recordings" / "sample.json.gz"
        return json.loads(gzip.decompress(path.read_bytes()))
    records = Path(records)
    paths = sorted(path for path in records.glob("*_race_*.json") if RACE_FILE.match(path.name))
    if not paths:
        raise ValueError(f"No race records found in {records}; run the live demo first")
    data = {
        "schema": 2, "models": [], "pairings": {}, "courses": {}, "races": {},
        "runtime": {}, "banks": {}, "unavailable": {}, "protocol": None,
        "replay_rules": {"stop_on_rock": True, "lane_move_seconds": 0.24},
        "presentation": "Measured replay of local model runs. Each runner stops on its first rock.",
        "provenance": {"kind": "live_run", "description":
            "Runs were measured separately with the same seeded course and aligned for replay. "
            "The game clock did not wait for model inference.", "sources": []},
    }
    for path in paths:
        record = _json(path)
        model, language, seed = RACE_FILE.match(path.name).groups()
        if abs(int(seed)) > 9007199254740991:
            raise ValueError("Recording seed must be within -9007199254740991 to 9007199254740991 for browser playback")
        if (record["model"], record["language"], str(record["seed"])) != (model, language, seed):
            raise ValueError(f"Filename and recording metadata disagree: {path.name}")
        course = record.pop("course")
        if seed in data["courses"] and data["courses"][seed] != course:
            raise ValueError(f"Cannot compare different courses for seed {seed}")
        profile = _profile(course)
        if data["protocol"] is not None and data["protocol"] != profile:
            raise ValueError("Cannot compare recordings with different course speed profiles")
        _validate_run(record, course)
        data["protocol"] = profile
        data["courses"][seed] = course
        # Exact wording/action order is reconstructed from the shared course and
        # fixed English/Vietnamese templates; full inputs remain in the raw file.
        record.pop("inputs", None)
        data["races"].setdefault(model, {}).setdefault(language, {})[seed] = record
        data["provenance"]["sources"].append({
            "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        runtime = records / f"{model}_runtime.json"
        if runtime.is_file():
            data["runtime"][model] = _json(runtime)
    data["models"] = sorted(data["races"], key=lambda model:
        (MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER), model))
    data["pairings"] = {language: [model for model in data["models"]
                                  if language in data["races"][model]]
                        for language in ("en", "vi")
                        if any(language in runs for runs in data["races"].values())}
    return data


def write_replay(data: dict, output: Path, *, template: Path | None = None,
                 background: Path | None = None) -> dict:
    """Embed data and artwork in a single HTML file, escaping script delimiters."""
    template = template or HERE / "web" / "template.html"
    background = background or HERE / "web" / "assets" / "world.png"
    html = template.read_text(encoding="utf-8")
    for marker in ("__GEM_DATA__", "__GEM_BACKGROUND__"):
        if html.count(marker) != 1:
            raise ValueError(f"Replay template must contain exactly one {marker} marker")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    artwork = "data:image/png;base64," + base64.b64encode(background.read_bytes()).decode("ascii")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace("__GEM_DATA__", payload).replace("__GEM_BACKGROUND__", artwork), encoding="utf-8")
    return {"output": str(output.resolve()), "bytes": output.stat().st_size,
            "models": data["models"], "languages": list(data["pairings"])}


def build_replay(records: Path | None, output: Path) -> dict:
    return write_replay(load_records(records), output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, help="Live-run directory; omit for bundled measured races")
    parser.add_argument("--output", type=Path, default=Path("gem-run.html"))
    parser.add_argument("--open", action="store_true", help="Open the generated file in a browser")
    args = parser.parse_args()
    try:
        result = build_replay(args.records, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Gem Run: {error}\n")
    print(json.dumps(result))
    if args.open:
        webbrowser.open(Path(result["output"]).as_uri())


if __name__ == "__main__":
    main()
