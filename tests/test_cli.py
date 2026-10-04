import json
import subprocess
import sys

from conftest import write_encoder_export


def test_cli_predict(tmp_path, modernbert_dir):
    path = write_encoder_export(tmp_path / "e", modernbert_dir, "option_query_v1")
    lines = [json.dumps({"state": "hi", "questions": {"n": {"type": "noul", "instructions": "Is it?"}}}),
             json.dumps({"state": "hi", "questions": {"n": {"type": "rank", "instructions": "x"}}})]
    p = subprocess.run([sys.executable, "-m", "quyet", "predict", str(path), "--device", "cpu"],
                       input="\n".join(lines) + "\n", capture_output=True, text=True)
    out = [json.loads(x) for x in p.stdout.strip().splitlines()]
    assert p.returncode == 1 and len(out) == 2
    assert out[0]["answers"]["n"]["type"] == "noul" and "unknown question type" in out[1]["error"]
