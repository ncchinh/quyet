import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from fnmatch import fnmatch
from functools import partial
from types import SimpleNamespace

import pytest


def runner():
    try:
        return importlib.import_module("examples.gem_run.run")
    except ModuleNotFoundError:
        pytest.fail("The portable Gem Run runner is not implemented")


def row(index, cells):
    return dict(id=index, start=float(index), end=float(index + 1), stage=0,
                goal="blue", variant=0, cells=cells, order=[0, 1, 2])


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TimedWorker:
    def __init__(self, clock, replies, *, fail=False):
        self.clock, self.replies, self.fail = clock, iter(replies), fail
        self.pending, self.sent, self.closed = None, [], False
        self.meta = {"backend": "fake", "warmup_calls": 3}

    def submit(self, row_id, language, state, questions):
        assert self.pending is None
        self.sent.append((row_id, self.clock(), language, state, questions))
        delay, lane = next(self.replies)
        self.pending = (self.clock() + delay, dict(row_id=row_id, lane=lane, answer={"action": "left"}))

    def poll(self, timeout=0):
        if self.fail and self.pending:
            raise EOFError("worker exited")
        if timeout:
            ready = self.pending[0] if self.pending else float("inf")
            self.clock.sleep(max(0, min(timeout, ready - self.clock())))
        return self.pending is not None and self.pending[0] <= self.clock()

    def receive(self):
        result = self.pending[1]
        self.pending = None
        return result

    def alive(self):
        return not self.fail or self.pending is None

    def close(self):
        self.closed = True


def test_live_loop_scores_prefix_and_never_submits_after_rock():
    clock = FakeClock()
    worker = TimedWorker(clock, [(.1, 0), (.2, 1)])
    data = runner().run_race(worker, "fake", "en", 701, rows=[
        row(0, ["blue", "red", "rock"]), row(1, ["blue", "rock", "red"]),
        row(2, ["blue", "red", "rock"])], clock=clock, sleep=clock.sleep)
    assert [request[0] for request in worker.sent] == [0, 1]
    assert worker.closed
    assert data["duration"] == 2
    assert data["observed_duration"] == pytest.approx(2)
    assert data["stop_reason"] == "collision"
    assert data["summary"]["timely_hits"] == 1
    assert data["summary"]["collisions"] == 1
    assert data["calls"][0]["issued"] == 0
    assert data["calls"][0]["received"] == pytest.approx(.1)
    assert data["calls"][0]["response_ms"] == pytest.approx(100)
    assert data["results"][0]["applied_at"] == pytest.approx(.1)
    assert data["rules"]["stop_on_first_rock"] is True
    json.dumps(data, allow_nan=False)


def test_hung_worker_does_not_pause_deadline_or_wait_for_final_drain():
    clock = FakeClock()
    worker = TimedWorker(clock, [(999, 0)])
    data = runner().run_race(worker, "fake", "vi", 701, rows=[
        row(0, ["blue", "rock", "red"]), row(1, ["blue", "red", "rock"])],
        clock=clock, sleep=clock.sleep)
    assert clock() == pytest.approx(101)
    assert data["duration"] == 1
    assert data["calls"] == []
    assert data["pending_request"] == {"row_id": 0, "issued": 0.0, "status": "cancelled_at_stop"}
    assert worker.closed and len(worker.sent) == 1


def test_late_worker_reply_is_logged_but_never_applied_to_next_row():
    clock = FakeClock()
    worker = TimedWorker(clock, [(2.2, 2), (.1, 0)])
    data = runner().run_race(worker, "fake", "en", 701,
        rows=[row(i, ["blue", "red", "rock"]) for i in range(3)],
        clock=clock, sleep=clock.sleep)
    assert [request[0] for request in worker.sent] == [0, 2]
    assert [r["lane"] for r in data["results"]] == [1, 1, 0]
    assert data["summary"]["timely_hits"] == 1
    assert data["summary"]["deadline_misses"] == 2
    assert data["calls"][0]["received"] == pytest.approx(2.2)


def test_worker_crash_preserves_results_and_continues_clock_until_rock():
    clock = FakeClock()
    worker = TimedWorker(clock, [(999, 0)], fail=True)
    data = runner().run_race(worker, "fake", "en", 701, rows=[
        row(0, ["blue", "red", "rock"]), row(1, ["blue", "rock", "red"])],
        clock=clock, sleep=clock.sleep)
    assert data["duration"] == 2 and len(data["results"]) == 2
    assert data["worker_error"]
    assert data["calls"][0]["error"]
    assert data["summary"]["deadline_misses"] == 2
    assert worker.closed and len(worker.sent) == 1


def test_keyboard_interrupt_always_closes_worker():
    clock = FakeClock()
    worker = TimedWorker(clock, [(999, 0)])
    worker.poll = lambda timeout=0: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        runner().run_race(worker, "fake", "en", 701,
            rows=[row(0, ["blue", "rock", "red"])], clock=clock, sleep=clock.sleep)
    assert worker.closed


def test_preemption_across_send_deadline_skips_unsent_row():
    class PreemptedClock(FakeClock):
        def __init__(self):
            super().__init__()
            self.reads = 0

        def __call__(self):
            self.reads += 1
            # The third read is the final deadline check before the first send.
            if self.reads == 3:
                self.now += 1.2
            return self.now

    clock = PreemptedClock()
    worker = TimedWorker(clock, [(.1, 0)])
    data = runner().run_race(worker, "fake", "en", 701,
        rows=[row(i, ["blue", "red", "rock"]) for i in range(2)],
        clock=clock, sleep=clock.sleep)
    assert [r[0] for r in worker.sent] == [1]
    assert data["issued_rows"] == [1]
    assert data["results"][0]["deadline_miss"] is True
    assert data["summary"]["timely_hits"] == 1


def test_reply_at_exact_deadline_is_late_in_live_loop():
    clock = FakeClock()
    worker = TimedWorker(clock, [(1, 0)])
    data = runner().run_race(worker, "fake", "en", 701,
        rows=[row(0, ["blue", "red", "rock"])], clock=clock, sleep=clock.sleep)
    assert data["calls"][0]["received"] == 1
    assert data["results"][0]["deadline_miss"] is True
    assert data["summary"]["timely_hits"] == 0


def native_response(choice="left"):
    return dict(model="fixture", answers={"move": {"type": "choice", "choice": choice,
        "probabilities": {"left": .8, "middle": .1, "right": .1}, "confidence": .8}},
        usage={"input_tokens": 40, "output_tokens": 0}, warnings=[])


class NativeBackend:
    meta = {"backend": "fixture"}

    def __init__(self, marker=None, fail_at=None):
        self.marker, self.fail_at, self.calls = marker, fail_at, 0

    def predict(self, state, questions):
        self.calls += 1
        if self.marker:
            with open(self.marker, "a") as stream:
                stream.write("predict\n")
        if self.calls == self.fail_at:
            raise ValueError("deliberate failure")
        return native_response()

    def close(self):
        if self.marker:
            with open(self.marker, "a") as stream:
                stream.write("closed\n")


def hang_on_start():
    time.sleep(60)


class HungBackend(NativeBackend):
    def predict(self, state, questions):
        time.sleep(60)


class CrashedBackend(NativeBackend):
    def predict(self, state, questions):
        os._exit(7)


def test_process_worker_warms_before_ready_and_gracefully_closes(tmp_path):
    marker = tmp_path / "worker.txt"
    worker = runner().ProcessWorker(partial(NativeBackend, marker=str(marker)), "en", warmup=2)
    assert marker.read_text().splitlines() == ["predict", "predict"]
    worker.submit(0, "en", "row", {"move": {"criteria": {"left": "L", "middle": "M", "right": "R"}}})
    assert worker.poll(3)
    answer = worker.receive()
    assert answer["row_id"] == 0 and answer["lane"] == 0
    assert answer["answer"]["action"] == "left"
    worker.close()
    assert not worker.alive()
    assert marker.read_text().splitlines()[-1] == "closed"


def test_worker_prediction_error_is_data_and_releases_the_request():
    worker = runner().ProcessWorker(partial(NativeBackend, fail_at=1), "en", warmup=0)
    try:
        worker.submit(0, "en", "row", {})
        assert worker.poll(3)
        response = worker.receive()
        assert response["row_id"] == 0 and response["lane"] is None
        assert "ValueError" in response["error"]
    finally:
        worker.close()


def test_failed_warmup_closes_loaded_backend(tmp_path):
    marker = tmp_path / "failed.txt"
    with pytest.raises(RuntimeError, match="ValueError"):
        runner().ProcessWorker(partial(NativeBackend, marker=str(marker), fail_at=1), "en", warmup=1)
    assert marker.read_text().splitlines()[-1] == "closed"


def test_startup_timeout_cleans_child_process():
    import multiprocessing
    before = {p.pid for p in multiprocessing.active_children()}
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        runner().ProcessWorker(hang_on_start, "en", startup_timeout=.05)
    assert time.monotonic() - start < 3
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_inflight_hang_is_killed_on_close():
    worker = runner().ProcessWorker(HungBackend, "en", warmup=0)
    worker.submit(0, "en", "row", {})
    start = time.monotonic()
    worker.close()
    assert time.monotonic() - start < 3
    assert not worker.alive()


def test_abrupt_child_exit_is_detected_and_cleaned():
    worker = runner().ProcessWorker(CrashedBackend, "en", warmup=0)
    started = time.monotonic()
    data = runner().run_race(worker, "fake", "en", 701, rows=[
        dict(row(0, ["blue", "rock", "red"]), end=.05)])
    assert time.monotonic() - started < 3
    assert data["worker_error"]
    assert data["duration"] == .05
    assert data["summary"]["collisions"] == 1
    assert not worker.alive()


@pytest.mark.parametrize("corrupt", ["choice", "keys", "nan", "sum", "warning", "truncated"])
def test_backend_rejects_invalid_native_answers(corrupt):
    backends = importlib.import_module("examples.gem_run.backends")
    result = native_response()
    answer = result["answers"]["move"]
    if corrupt == "choice":
        answer["choice"] = "fly"
    elif corrupt == "keys":
        answer["probabilities"]["fly"] = answer["probabilities"].pop("left")
    elif corrupt == "nan":
        answer["probabilities"]["left"] = float("nan")
    elif corrupt == "sum":
        answer["probabilities"]["left"] = .1
    elif corrupt == "warning":
        result["warnings"] = [{"code": "state_truncated"}]
    else:
        answer["truncated"] = True
    with pytest.raises(ValueError):
        backends.normalize_answer(result, "en")


def test_native_confidence_is_preserved_without_changing_its_meaning():
    backends = importlib.import_module("examples.gem_run.backends")
    result = native_response()
    result["answers"]["move"]["confidence"] = .62
    answer = backends.normalize_answer(result, "en")
    assert answer["confidence"] == .62
    assert answer["probabilities"]["left"] == .8


def test_cli_chooses_language_matched_models_and_pins_laya():
    module = runner()
    args = module.make_parser().parse_args(["--compare-laya", "--device", "cpu"])
    specs = module.model_specs(args)
    assert [(s.name, s.language) for s in specs] == [
        ("small_en", "en"), ("laya_english", "en"),
        ("small", "vi"), ("laya_multilingual", "vi")]
    assert specs[0].source == "chinhnc/Quyet-1.0-Small-EN"
    assert specs[2].source == "chinhnc/Quyet-1.0-Small"
    assert specs[1].revision == "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    assert specs[3].subfolder == "multilingual"
    assert args.seeds == "701"
    assert [s.portable_metadata()["display_name"] for s in specs] == [
        "Quyet Small-EN", "Laya EN", "Quyet Small", "Laya Multilingual"]


def test_custom_model_paths_are_not_exposed_in_runtime_metadata(tmp_path):
    backends = importlib.import_module("examples.gem_run.backends")
    model_path = tmp_path / "private" / "model"
    model_path.mkdir(parents=True)
    spec = backends.ModelSpec("custom", "quyet", str(model_path), "en")
    metadata = spec.portable_metadata()
    assert metadata["source"] == {"kind": "local", "name": "model"}
    assert metadata["display_name"] == "model"
    assert str(tmp_path) not in json.dumps(metadata)


def fake_sdk_modules(monkeypatch):
    events = []
    torch = SimpleNamespace(set_num_threads=lambda n: events.append(("threads", n)),
        __version__="fixture", cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return events


def test_quyet_adapter_uses_public_load_and_strict_predict(monkeypatch):
    backends = importlib.import_module("examples.gem_run.backends")
    events = fake_sdk_modules(monkeypatch)

    class Model:
        def predict(self, state, questions, **kwargs):
            events.append(("predict", state, questions, kwargs))
            return native_response()

    def load(source, **kwargs):
        events.append(("load", source, kwargs))
        return Model()

    monkeypatch.setitem(sys.modules, "quyet", SimpleNamespace(load=load, __version__="fixture"))
    spec = backends.ModelSpec("small_en", "quyet", "chinhnc/Quyet-1.0-Small-EN", "en", revision="pin")
    backend = backends.PublicBackend(spec, device="auto", threads=2)
    output = backend.predict("visible row", {"move": {"type": "choice"}})
    assert output["answers"]["move"]["choice"] == "left"
    assert ("load", "chinhnc/Quyet-1.0-Small-EN", {"device": "cpu", "revision": "pin"}) in events
    assert events[-1] == ("predict", "visible row", {"move": {"type": "choice"}}, {"strict": True})
    assert backend.meta["device"] == "cpu" and backend.meta["threads"] == 2


@pytest.mark.parametrize("subfolder", [None, "multilingual"])
def test_laya_adapter_resolves_pin_before_loading_explicit_checkpoint(monkeypatch, tmp_path, subfolder):
    backends = importlib.import_module("examples.gem_run.backends")
    events = fake_sdk_modules(monkeypatch)

    def snapshot_download(source, **kwargs):
        events.append(("download", source, kwargs))
        prefix = f"{subfolder}/" if subfolder else ""
        required = [prefix + f for f in ("rl_agent_config.json", "model.safetensors",
                    "tokenizer/tokenizer.json", "encoder/config.json")]
        for filename in required:
            selected = any(fnmatch(filename, p) for p in kwargs["allow_patterns"])
            ignored = any(fnmatch(filename, p) for p in (kwargs.get("ignore_patterns") or []))
            assert selected and not ignored, f"Required Laya file was not downloaded: {filename}"
        return str(tmp_path)

    def load(source, **kwargs):
        events.append(("load", source, kwargs))
        return NativeBackend()

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(load=load, __version__="fixture"))
    spec = backends.ModelSpec("laya_multilingual", "laya", "convaiinnovations/laya", "vi",
        revision="public-pin", subfolder=subfolder)
    backend = backends.PublicBackend(spec, device="cpu")
    assert events[1][0:2] == ("download", "convaiinnovations/laya")
    assert events[1][2]["revision"] == "public-pin"
    assert events[2] == ("load", str(tmp_path), {"subfolder": subfolder, "device": "cpu", "fast": False})
    assert str(tmp_path) not in json.dumps(backend.meta)


def test_laya_context_limit_is_not_silently_treated_as_valid_input(monkeypatch, tmp_path):
    backends = importlib.import_module("examples.gem_run.backends")
    fake_sdk_modules(monkeypatch)
    model = NativeBackend()
    model.cfg = {"max_len": 40}
    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(load=lambda *a, **kw: model, __version__="fixture"))
    spec = backends.ModelSpec("laya_english", "laya", str(tmp_path), "en")
    backend = backends.PublicBackend(spec, device="cpu")
    with pytest.raises(ValueError, match="context limit"):
        backend.predict("row", {"move": {}})


def test_help_runs_without_installed_model_packages():
    result = subprocess.run([sys.executable, "-S", "-m", "examples.gem_run.run", "--help"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert "--compare-laya" in result.stdout


def cli_backend(spec, device, threads):
    return NativeBackend()


def failing_cli_backend(spec, device, threads):
    raise FileNotFoundError("private model location")


def test_cli_startup_failure_reports_useful_cause_without_private_sdk_text(monkeypatch, tmp_path):
    module = runner()
    monkeypatch.setattr(module, "PublicBackend", failing_cli_backend)
    result = module.main(["--language", "en", "--out", str(tmp_path)])
    assert result == 1
    failure = json.loads((tmp_path / "small_en_unavailable_en_701.json").read_text())
    assert "FileNotFoundError" in failure["error"]
    assert "private model location" not in json.dumps(failure)


@pytest.mark.parametrize("language", ["en", "vi"])
@pytest.mark.parametrize("artifact", [
    "small_en_race_en_701.json", "protocol.json", "small_en_runtime.json",
    "small_en_unavailable_en_701.json", "gem-run.html",
])
def test_cli_rejects_prior_recordings_before_failed_or_changed_language_rerun(
        monkeypatch, tmp_path, capsys, artifact, language):
    module = runner()
    original = {artifact: b"previous recording", "notes.txt": b"keep this too"}
    for name, content in original.items():
        (tmp_path / name).write_bytes(content)
    starts, builds = [], []

    def failed_worker(*args, **kwargs):
        starts.append(True)
        raise RuntimeError("a retry would fail to load")

    monkeypatch.setattr(module, "ProcessWorker", failed_worker)
    monkeypatch.setitem(sys.modules, "examples.gem_run.replay", SimpleNamespace(
        build_replay=lambda *args: builds.append(True)))
    with pytest.raises(SystemExit) as error:
        module.main(["--language", language, "--out", str(tmp_path)])
    assert error.value.code == 2
    assert "new --out" in capsys.readouterr().err
    assert starts == [] and builds == []
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original


def test_cli_writes_records_runtime_and_viewable_replay(monkeypatch, tmp_path):
    module = runner()
    monkeypatch.setattr(module, "PublicBackend", cli_backend)
    (tmp_path / "notes.txt").write_text("unrelated user file")

    def build_replay(records, output):
        paths = list(records.glob("*_race_*.json"))
        assert len(paths) == 1
        output.write_text("<!doctype html><title>Recorded race</title>")
        return {"path": str(output)}

    monkeypatch.setitem(sys.modules, "examples.gem_run.replay", SimpleNamespace(build_replay=build_replay))
    result = module.main(["--language", "en", "--stages", "1", "--stage-seconds", "1",
                          "--warmup", "1", "--device", "cpu", "--out", str(tmp_path)])
    assert result == 0
    data = json.loads((tmp_path / "small_en_race_en_701.json").read_text())
    assert data["duration"] == 1
    assert data["runtime"]["warmup_calls"] == 1
    assert json.loads((tmp_path / "small_en_runtime.json").read_text())["backend"] == "fixture"
    assert (tmp_path / "gem-run.html").is_file()
    assert (tmp_path / "notes.txt").read_text() == "unrelated user file"


@pytest.mark.parametrize("seed", [9007199254740992, -9007199254740992])
def test_cli_rejects_seeds_outside_browser_integer_range(tmp_path, capsys, monkeypatch, seed):
    module = runner()

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid seeds must be rejected before any model starts")

    monkeypatch.setattr(module, "ProcessWorker", forbidden)

    with pytest.raises(SystemExit) as error:
        module.main(["--seeds", str(seed), "--out", str(tmp_path / "output")])
    assert error.value.code == 2
    assert "9007199254740991" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
