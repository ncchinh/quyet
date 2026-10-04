import pytest

import quyet
from quyet.errors import QuestionError
from conftest import write_encoder_export

HEADS = ["option_query_v1", "refine_pool_v1", "option_query_wide"]
QUESTIONS = {
    "c": {"type": "choice", "instructions": "What does the user want?", "criteria": {"cancel": "close", "limit": None}},
    "s": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
    "n": {"type": "noul", "instructions": "Is the user angry?"},
}


@pytest.fixture(scope="module", params=HEADS)
def enc(request, tmp_path_factory, modernbert_dir):
    path = write_encoder_export(tmp_path_factory.mktemp("enc") / request.param, modernbert_dir, request.param)
    return quyet.load(str(path), device="cpu")


def test_predict_shape(enc):
    r = enc.predict({"msg": "please close my card"}, QUESTIONS)
    assert r["model"] == enc.name and list(r["answers"]) == ["c", "s", "n"]
    c, s, n = r["answers"]["c"], r["answers"]["s"], r["answers"]["n"]
    assert c["choice"] in ("cancel", "limit") and abs(sum(c["probabilities"].values()) - 1) < 1e-3
    assert 0 <= s["score"] <= 2 and set(s["probabilities"]) == {"0", "1", "2"}
    raw = enc.raw_probabilities([{"state": {"msg": "please close my card"}, "questions": QUESTIONS}])[0]
    assert n["noul"] == round(raw["n"][0], 4) and abs(sum(raw["n"]) - 1) < 1e-9
    assert r["usage"]["input_tokens"] > 0 and r["usage"]["output_tokens"] == 0 and r["warnings"] == []


def test_batch_invariance(enc):
    reqs = [{"state": "short", "questions": QUESTIONS},
            {"state": "a considerably longer state " * 20, "questions": {"n": QUESTIONS["n"]}},
            {"state": [{"role": "user", "content": "hi"}], "questions": {"c": QUESTIONS["c"]}}]
    together = enc.raw_probabilities(reqs)
    for req, got in zip(reqs, together):
        alone = enc.raw_probabilities([req])[0]
        for qid in alone:
            assert max(abs(a - b) for a, b in zip(alone[qid], got[qid])) < 1e-4


def test_temperatures_apply(enc):
    req = [{"state": "x", "questions": {"c": QUESTIONS["c"]}}]
    base = max(enc.raw_probabilities(req)[0]["c"])
    enc.temps = {"choice:2": 1e-6}
    sharp = max(enc.raw_probabilities(req)[0]["c"])
    enc.temps = {}
    assert sharp > base and sharp > 0.99


def test_truncation_warning_and_strict(tmp_path, modernbert_dir):
    m = quyet.load(str(write_encoder_export(tmp_path / "e", modernbert_dir, "option_query_v1", max_len=48)),
                   device="cpu")
    r = m.predict("word " * 200, {"n": QUESTIONS["n"]})
    assert r["answers"]["n"]["truncated"] is True
    w = r["warnings"][0]
    assert w["code"] == "state_truncated" and w["kept"] == "head" and w["kept_tokens"] < w["state_tokens"]
    with pytest.raises(QuestionError, match="strict"):
        m.predict("word " * 200, {"n": QUESTIONS["n"]}, strict=True)


def test_bad_state_type(enc):
    with pytest.raises(QuestionError, match="state must be"):
        enc.predict(42, QUESTIONS)
