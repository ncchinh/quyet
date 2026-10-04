from quyet.answers import build_response, format_answer
from quyet.questions import parse_question


def test_choice_answer():
    d = parse_question({"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": "B"}})
    assert format_answer(d, [0.25, 0.75]) == {"type": "choice", "choice": "b", "confidence": 0.75,
                                              "probabilities": {"a": 0.25, "b": 0.75}}


def test_score_answer_uses_expected_level_and_legend():
    d = parse_question({"type": "score", "instructions": "Rate", "criteria": ["low", None, "high"]})
    a = format_answer(d, [0.2, 0.3, 0.5], truncated=True)
    assert a == {"type": "score", "score": 1.3, "confidence": 0.5, "legend": {"0": "low", "1": "1", "2": "high"},
                 "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}, "truncated": True}


def test_noul_answer_is_p_true():
    d = parse_question({"type": "noul", "instructions": "Is it?"})
    assert format_answer(d, [0.123456, 0.876544]) == {"type": "noul", "noul": 0.1235, "confidence": 0.8765}


def test_build_response_envelope():
    d = parse_question({"type": "noul", "instructions": "Is it?"})
    r = build_response("m", {"q": d}, {"q": [0.9, 0.1]}, {"q": False}, 42, [{"code": "x"}])
    assert r == {"model": "m", "answers": {"q": {"type": "noul", "noul": 0.9, "confidence": 0.9}},
                 "usage": {"input_tokens": 42, "output_tokens": 0}, "warnings": [{"code": "x"}]}
