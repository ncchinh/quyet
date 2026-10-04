"""Run real-clock Gem Run races with public Quyet and optional Laya SDKs.

From a checkout: python -m examples.gem_run.run --language en --device cpu
The game runs in the parent; a disposable spawned process owns each model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import multiprocessing as mp
from pathlib import Path
import re
import time

from .backends import LAYA_REPO, LAYA_REVISION, ModelSpec, PublicBackend, normalize_answer
from .engine import LANGUAGES, SCHEMA_VERSION, Race, course, lane_for, request_for, rules, summarize


DEFAULT_STARTUP_TIMEOUT = 900


class WorkerStartupError(RuntimeError):
    """A startup failure whose message was scrubbed by the child process."""


def _error(exc, phase):
    # Do not put SDK exceptions containing local paths or credentials in a replay.
    if isinstance(exc, WorkerStartupError):
        return str(exc)
    return f"{type(exc).__name__} during {phase}"


def _worker(connection, factory, language, warmup):
    backend = None
    ready = False
    try:
        backend = factory()
        warmup_rows = course(9901, stages=3)
        for index in range(warmup):
            state, questions = request_for(warmup_rows[index % len(warmup_rows)], language)
            normalize_answer(backend.predict(state, questions), language)
        connection.send(dict(ready=True, meta=dict(getattr(backend, "meta", {}), warmup_calls=warmup)))
        ready = True
        while True:
            message = connection.recv()
            if message is None:
                break
            row_id, language, state, questions = message
            try:
                started = time.monotonic()
                answer = normalize_answer(backend.predict(state, questions), language)
                answer["latency_ms"] = (time.monotonic() - started) * 1000
                response = dict(row_id=row_id, lane=lane_for(answer["action"], language), answer=answer)
            except Exception as exc:
                response = dict(row_id=row_id, lane=None, answer=None, error=_error(exc, "prediction"))
            connection.send(response)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        if not ready:
            try:
                connection.send(dict(ready=False, error=_error(exc, "model loading or warmup")))
            except (EOFError, OSError):
                pass
    finally:
        if backend is not None:
            close = getattr(backend, "close", None)
            if close is not None:
                close()
        connection.close()


class ProcessWorker:
    """One request at a time; bounded startup and shutdown even if inference hangs."""

    def __init__(self, factory, language, *, warmup=5, startup_timeout=DEFAULT_STARTUP_TIMEOUT):
        self.pending = False
        self.closed = False
        self.meta = {}
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, factory, language, warmup), daemon=True)
        try:
            self.process.start()
            child.close()
            if not self.connection.poll(startup_timeout):
                raise TimeoutError("Model loading or warmup timed out")
            ready = self.connection.recv()
            if not ready.get("ready"):
                raise WorkerStartupError(ready.get("error", "Model worker failed to start"))
            self.meta = ready["meta"]
        except BaseException:
            child.close()
            self.close()
            raise

    def submit(self, row_id, language, state, questions):
        if self.pending:
            raise RuntimeError("A request is already in flight")
        self.connection.send((row_id, language, state, questions))
        self.pending = True

    def poll(self, timeout=0):
        return self.connection.poll(timeout)

    def receive(self):
        response = self.connection.recv()
        self.pending = False
        return response

    def alive(self):
        return self.process.pid is not None and self.process.is_alive()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.alive() and not self.pending:
                try:
                    self.connection.send(None)
                except (EOFError, OSError):
                    pass
                self.process.join(.2)
            if self.alive():
                self.process.terminate()
                self.process.join(1)
            if self.alive():
                self.process.kill()
                self.process.join(1)
            if self.process.pid is not None:
                self.process.join(0)
        finally:
            self.connection.close()


def run_race(worker, name, language, seed, stages=9, *, stage_seconds=4,
             rows=None, clock=time.monotonic, sleep=time.sleep):
    """Consume a ready worker, record actual receive times, and always close it.

    The injected worker supports submit/poll/receive/alive/close. A supplied clock
    and sleep let tests exercise deadlines without models, a GPU or real waiting.
    Unanswered requests are recorded separately and cancelled at the terminal row;
    there is no final inference drain that could hang or leak into another race.
    """
    calls, issued_times = [], {}
    worker_error = None
    try:
        if language not in LANGUAGES:
            raise ValueError("language must be en or vi")
        rows = course(seed, stages, stage_seconds) if rows is None else rows
        inputs = [dict(zip(("state", "questions"), request_for(row, language))) for row in rows]
        by_id = {row["id"]: (row, payload) for row, payload in zip(rows, inputs)}
        race = Race(rows)
        epoch = clock()

        def failed_response(exc):
            nonlocal worker_error
            worker_error = _error(exc, "worker communication")
            if race.busy is None:
                return None
            received = clock() - epoch
            issued = issued_times[race.busy]
            response = dict(row_id=race.busy, lane=None, answer=None, error=worker_error,
                            issued=issued, received=received, response_ms=(received - issued) * 1000)
            calls.append(response)
            return response

        while not race.finished:
            response = None
            if worker_error is None:
                try:
                    if worker.poll():
                        response = worker.receive()
                        if not isinstance(response, dict) or response.get("row_id") != race.busy:
                            raise ValueError("Worker returned an unsolicited response")
                        received = clock() - epoch
                        issued = issued_times[response["row_id"]]
                        response = dict(response, issued=issued, received=received,
                                        response_ms=(received - issued) * 1000)
                        calls.append(response)
                    elif not worker.alive():
                        raise EOFError("Worker exited")
                except (EOFError, OSError, ValueError) as exc:
                    response = failed_response(exc)
            now = clock() - epoch
            issued_id = race.tick(now, response, allow_request=worker_error is None)
            if issued_id is not None:
                row, payload = by_id[issued_id]
                issued_at = clock() - epoch
                if issued_at >= row["end"]:
                    race.cancel_unsent(issued_id)
                    continue
                issued_times[issued_id] = issued_at
                race.issued_at[issued_id] = issued_at
                try:
                    worker.submit(issued_id, language, payload["state"], payload["questions"])
                except (EOFError, OSError) as exc:
                    response = failed_response(exc)
                    race.tick(clock() - epoch, response, allow_request=False)
            if not race.finished:
                row = rows[race.index]
                next_event = row["start"] if now < row["start"] else row["end"]
                remaining = next_event - (clock() - epoch)
                if remaining > 0:
                    if race.busy is not None and worker_error is None:
                        try:
                            worker.poll(min(remaining, .005))
                        except (EOFError, OSError) as exc:
                            response = failed_response(exc)
                            race.tick(clock() - epoch, response, allow_request=False)
                    else:
                        sleep(min(remaining, .002))
        observed_duration = clock() - epoch
        pending = None if race.busy is None else dict(
            row_id=race.busy, issued=issued_times[race.busy], status="cancelled_at_stop")
        rule_data = rules(stages, stage_seconds)
        rule_data["planned_duration"] = rows[-1]["end"] if rows else 0
        return dict(schema_version=SCHEMA_VERSION, model=name, language=language, seed=seed,
                    course=rows, inputs=inputs, results=race.results, calls=calls,
                    issued_rows=race.issued_rows, duration=race.stopped_at,
                    observed_duration=observed_duration, stop_reason=race.stop_reason,
                    pending_request=pending, worker_error=worker_error, rules=rule_data,
                    runtime=dict(getattr(worker, "meta", {})), summary=summarize(rows, race.results, calls))
    finally:
        worker.close()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "vi", "all"), default="all")
    parser.add_argument("--seeds", default="701", help="Comma-separated integer course seeds")
    parser.add_argument("--stages", type=int, default=9, help="Speed stages, 1–9")
    parser.add_argument("--stage-seconds", type=int, default=4, help="Seconds per stage, 1–60")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model", help="Quyet repository id or local directory; defaults to Small-EN / Small")
    parser.add_argument("--revision", help="Optional Quyet repository revision")
    parser.add_argument("--compare-laya", action="store_true", help="Run the matching Laya checkpoint separately")
    parser.add_argument("--laya-model", default=LAYA_REPO, help="Laya repository id or local bundle directory")
    parser.add_argument("--laya-revision", default=LAYA_REVISION, help="Pinned Laya repository revision")
    parser.add_argument("--laya-subfolder", help="Override checkpoint subfolder; default root for EN, multilingual for VI")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup calls before starting each race clock")
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT,
                        help="Seconds allowed for model download/loading and warmup (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=Path("gem-run-output"),
                        help="Output directory with no previous Gem Run artifacts")
    parser.add_argument("--open", action="store_true", help="Open the generated standalone HTML replay")
    return parser


def model_specs(args):
    specs = []
    languages = LANGUAGES if args.language == "all" else (args.language,)
    for language in languages:
        name = "small_en" if language == "en" else "small"
        source = "chinhnc/Quyet-1.0-Small-EN" if language == "en" else "chinhnc/Quyet-1.0-Small"
        if args.model:
            source = args.model
            name = "quyet_" + (re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source).name).strip("_") or "custom")
        specs.append(ModelSpec(name, "quyet", source, language, revision=args.revision))
        if args.compare_laya:
            subfolder = args.laya_subfolder if args.laya_subfolder is not None else (
                "multilingual" if language == "vi" else None)
            specs.append(ModelSpec("laya_english" if language == "en" else "laya_multilingual",
                "laya", args.laya_model, language, revision=args.laya_revision, subfolder=subfolder or None))
    return specs


def _save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        seeds = list(dict.fromkeys(int(seed.strip()) for seed in args.seeds.split(",")))
        if any(abs(seed) > 9007199254740991 for seed in seeds):
            raise ValueError("seeds must be within -9007199254740991 to 9007199254740991 for browser playback")
        course(seeds[0], args.stages, args.stage_seconds)
        if args.threads < 1 or args.warmup < 0 or not 0 < args.startup_timeout < float("inf"):
            raise ValueError("threads/startup-timeout must be positive and warmup nonnegative")
    except (ValueError, IndexError) as exc:
        parser.error(str(exc))
    artifacts = ("protocol.json", "gem-run.html", "*_race_*.json", "*_runtime.json", "*_unavailable*.json")
    if any(next(args.out.glob(pattern), None) is not None for pattern in artifacts):
        parser.error("Output directory already contains Gem Run artifacts; use a new --out directory. "
                     "Existing recordings have been left unchanged.")
    specs = model_specs(args)
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = dict(schema_version=SCHEMA_VERSION, purpose="Gem Run release example; not a general model ranking",
                    created_utc=datetime.now(timezone.utc).isoformat(), actual_seeds=seeds,
                    actual_models=[spec.portable_metadata() for spec in specs],
                    actual_stages=args.stages, duration_seconds=args.stages * args.stage_seconds,
                    **rules(args.stages, args.stage_seconds))
    _save(args.out / "protocol.json", protocol)
    completed, errors = 0, 0
    for spec in specs:
        for seed in seeds:
            print(json.dumps(dict(loading=spec.name, language=spec.language, seed=seed)), flush=True)
            try:
                worker = ProcessWorker(partial(PublicBackend, spec, args.device, args.threads), spec.language,
                                       warmup=args.warmup, startup_timeout=args.startup_timeout)
                data = run_race(worker, spec.name, spec.language, seed, args.stages,
                                stage_seconds=args.stage_seconds)
                _save(args.out / f"{spec.name}_runtime.json", data["runtime"])
                _save(args.out / f"{spec.name}_race_{spec.language}_{seed}.json", data)
                completed += 1
                if data["worker_error"]:
                    errors += 1
                print(json.dumps(dict(model=spec.name, language=spec.language, seed=seed,
                    duration=data["duration"], stop_reason=data["stop_reason"], **data["summary"])), flush=True)
            except Exception as exc:
                errors += 1
                error = dict(model=spec.name, language=spec.language, seed=seed,
                             status="unavailable", error=_error(exc, "race startup or recording"))
                _save(args.out / f"{spec.name}_unavailable_{spec.language}_{seed}.json", error)
                print(json.dumps(error), flush=True)
    if completed:
        from .replay import build_replay

        output = args.out / "gem-run.html"
        build_replay(args.out, output)
        print(f"Replay: {output.resolve()}", flush=True)
        if args.open:
            import webbrowser
            webbrowser.open(output.resolve().as_uri())
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
