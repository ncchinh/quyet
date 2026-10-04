"""quyet predict MODEL: JSON lines {"state", "questions"} in, one JSON response (or {"error"}) per line out."""
from __future__ import annotations

import argparse
import json
import sys

from .errors import QuestionError
from .hub import load


def main(argv=None):
    ap = argparse.ArgumentParser(prog="quyet", description="Run a Quyet decision model.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="answer JSON-lines requests {state, questions}")
    p.add_argument("model", help="Hugging Face repo id (e.g. chinhnc/Quyet-1.0-Small) or local directory")
    p.add_argument("--input", help="JSON-lines file (default: stdin)")
    p.add_argument("--device", help="cpu, cuda, cuda:1, ... (default: cuda if available)")
    p.add_argument("--strict", action="store_true", help="fail instead of truncating a long state")
    a = ap.parse_args(argv)
    model = load(a.model, device=a.device)
    failed = False
    src = open(a.input, encoding="utf8") if a.input else sys.stdin
    with src:
        for line in src:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                out = model.predict(req["state"], req["questions"], strict=a.strict)
            except (QuestionError, KeyError, TypeError, json.JSONDecodeError) as e:
                failed = True
                out = {"error": f"{type(e).__name__}: {e}"}
            print(json.dumps(out, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
