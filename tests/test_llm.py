import pytest
import torch

import quyet
from quyet.errors import QuestionError
from quyet.llm import prompt as P
from quyet.questions import parse_question, parse_questions
from conftest import write_llm_export

QUESTIONS = {
    "c": {"type": "choice", "instructions": "What does the user want?", "criteria": {"cancel": "close", "limit": None}},
    "s": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
    "n": {"type": "noul", "instructions": "Is the user angry?", "criteria": {"true": "angry"}},
}


@pytest.fixture(scope="module")
def llm(tmp_path_factory, qwen_tok_dir):
    path = write_llm_export(tmp_path_factory.mktemp("llm") / "tiny", qwen_tok_dir)
    return quyet.load(str(path), device="cpu", dtype=torch.float32)


def test_prompt_text(llm):
    d = parse_question(QUESTIONS["n"])
    text = P.build_prompt(llm.tok, "sky is grey", d)
    assert P.SYSTEM in text and "State:\nsky is grey\n\nQuestion: Is the user angry?\n" in text
    assert "Choose A if the statement is true for this state, B if it is not." in text
    assert "Options:\nA. angry\nB. false\n\nAnswer with one letter." in text
    assert P.option_texts(parse_question(QUESTIONS["c"])) == ["close", "limit"]
    assert P.option_texts(parse_question({"type": "score", "instructions": "r", "criteria": [None, "hi"]})) == ["0", "hi"]
    assert P.serialize_state({"a": 1}) == '{\n "a": 1\n}'
    assert len(P.letter_ids(llm.tok)) == 10


def test_llm_predict(llm):
    r = llm.predict({"msg": "close my card now!"}, QUESTIONS)
    assert list(r["answers"]) == ["c", "s", "n"]
    assert abs(sum(r["answers"]["c"]["probabilities"].values()) - 1) < 1e-3
    raw = llm.raw_probabilities([{"state": {"msg": "close my card now!"}, "questions": QUESTIONS}])[0]
    assert r["answers"]["n"]["noul"] == round(raw["n"][0], 4)          # A = true
    assert r["usage"]["input_tokens"] > 0 and r["warnings"] == []


def test_llm_temperatures(llm):
    req = [{"state": "x", "questions": {"s": QUESTIONS["s"]}}]
    base = max(llm.raw_probabilities(req)[0]["s"])
    llm.temps = {"score": 1e-6}
    sharp = max(llm.raw_probabilities(req)[0]["s"])
    llm.temps = {}
    assert sharp > base and sharp > 0.99


def test_llm_batch_invariance(llm):
    reqs = [{"state": "short", "questions": QUESTIONS},
            {"state": "a considerably longer state " * 30, "questions": {"c": QUESTIONS["c"]}}]
    together = llm.raw_probabilities(reqs)
    for req, got in zip(reqs, together):
        alone = llm.raw_probabilities([req])[0]
        for qid in alone:
            assert max(abs(a - b) for a, b in zip(alone[qid], got[qid])) < 1e-4


def test_list_state_truncation_keeps_tail(llm):
    llm.max_prompt_tokens, llm.min_state_tokens = 200, 8
    try:
        state = [{"role": "user", "content": "first " * 300}, {"role": "user", "content": "LAST TURN"}]
        r = llm.predict(state, {"n": QUESTIONS["n"]})
        assert r["answers"]["n"]["truncated"] is True and r["warnings"][0]["kept"] == "tail"
        text, n, truncated = P.truncate_state(llm.tok, state, 20)
        assert truncated and text.startswith("… ") and "LAST TURN" in text
        with pytest.raises(QuestionError, match="strict"):
            llm.predict(state, {"n": QUESTIONS["n"]}, strict=True)
        llm.max_prompt_tokens = 60
        with pytest.raises(QuestionError, match="no room for the state"):
            llm.predict(state, {"n": QUESTIONS["n"]})
    finally:
        llm.max_prompt_tokens, llm.min_state_tokens = 8000, 256


def test_device_map_passthrough(tmp_path, qwen_tok_dir):
    pytest.importorskip("accelerate")
    path = write_llm_export(tmp_path / "tiny", qwen_tok_dir)
    m = quyet.load(str(path), device_map="cpu", dtype=torch.float32)
    assert m.device == "cpu"
    r = m.predict("hello", {"n": QUESTIONS["n"]})
    assert abs(r["answers"]["n"]["noul"] + (1 - r["answers"]["n"]["noul"]) - 1) < 1e-9


def test_device_map_rejected_for_encoders(tmp_path, modernbert_dir):
    from conftest import write_encoder_export
    path = write_encoder_export(tmp_path / "e", modernbert_dir, "option_query_v1")
    with pytest.raises(ValueError, match="device_map"):
        quyet.load(str(path), device_map="auto")


def test_prompt_v2_is_the_diet_prompt(llm):
    """prompt_version 2 (Quyet-1.0-Large): no system message, no closing line, compact JSON; trained instructions kept."""
    d = parse_question(QUESTIONS["n"])
    text = P.build_prompt(llm.tok, "sky is grey", d, version=2)
    assert P.SYSTEM not in text and "Answer with one letter." not in text
    assert "State:\nsky is grey\n\nQuestion: Is the user angry?\n" in text
    assert "Choose A if the statement is true for this state, B if it is not." in text
    assert "Options:\nA. angry\nB. false" in text
    assert P.serialize_state({"a": [1, 2]}, compact=True) == '{"a":[1,2]}'
    v1 = P.build_prompt(llm.tok, "sky is grey", d)
    assert len(llm.tok(text)["input_ids"]) < len(llm.tok(v1)["input_ids"])


def test_runtime_uses_config_prompt_version(tmp_path, qwen_tok_dir):
    path = write_llm_export(tmp_path / "v2", qwen_tok_dir, prompt_version=2)
    m = quyet.load(str(path), device="cpu", dtype=torch.float32)
    ids, _, _ = m._prompt_ids({"a": 1}, parse_questions({"q": QUESTIONS["n"]}), strict=False)
    text = m.tok.decode(ids[0])
    assert P.SYSTEM not in text and '{"a":1}' in text


def test_unknown_prompt_version_rejected(tmp_path, qwen_tok_dir):
    path = write_llm_export(tmp_path / "v9", qwen_tok_dir, prompt_version=9)
    with pytest.raises(Exception, match="prompt_version"):
        quyet.load(str(path), device="cpu", dtype=torch.float32)
