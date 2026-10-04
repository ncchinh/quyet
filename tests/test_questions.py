import pytest

from quyet.errors import QuestionError
from quyet.questions import MAX_OPTIONS, Decision, describe, parse_question, parse_questions


def test_choice_dict_keeps_order_and_texts():
    d = parse_question({"type": "choice", "instructions": "Pick", "criteria": {"b": "bee", "a": None, "c": ""}})
    assert d == Decision("choice", "Pick", ("b", "a", "c"), ("bee", None, None))


def test_choice_list_has_no_texts():
    d = parse_question({"type": "choice", "instructions": "Pick", "criteria": ["x", "y"]})
    assert d.labels == ("x", "y") and d.texts == (None, None)


def test_score_labels_are_level_indices():
    d = parse_question({"type": "score", "instructions": "Rate", "criteria": ["low", None, "high"]})
    assert d.labels == ("0", "1", "2") and d.texts == ("low", None, "high")


def test_noul_order_is_true_false_and_criteria_optional():
    d = parse_question({"type": "noul", "instructions": "Is it?"})
    assert d.labels == ("true", "false") and d.texts == (None, None)
    d = parse_question({"type": "noul", "instructions": "Is it?", "criteria": {"false": "no", "true": "yes"}})
    assert d.texts == ("yes", "no")


def test_describe_non_string_and_empty():
    assert describe(None) is None and describe("") is None
    assert describe("plain") == "plain"
    assert describe({"a": 1, "b": [1, 2]}) == '{"a": 1, "b": [1, 2]}'
    assert describe({"tên": "Hà Nội"}) == '{"tên": "Hà Nội"}'
    assert describe(3) == "3"


def test_option_count_limits():
    with pytest.raises(QuestionError, match="at least 2"):
        parse_question({"type": "choice", "instructions": "Pick", "criteria": {"only": "one"}})
    many = {str(i): str(i) for i in range(MAX_OPTIONS + 1)}
    with pytest.raises(QuestionError, match="at most 10"):
        parse_question({"type": "choice", "instructions": "Pick", "criteria": many})
    with pytest.raises(QuestionError, match="at most 10"):
        parse_question({"type": "score", "instructions": "Rate", "criteria": [str(i) for i in range(11)]})


@pytest.mark.parametrize("q, msg", [
    ({"type": "rank", "instructions": "x", "criteria": ["a", "b"]}, "unknown question type"),
    ({"type": "choice", "instructions": "  ", "criteria": ["a", "b"]}, "non-empty"),
    ({"type": "choice", "instructions": "x", "criteria": "a,b"}, "choice criteria"),
    ({"type": "choice", "instructions": "x", "criteria": ["a", "a"]}, "unique"),
    ({"type": "score", "instructions": "x", "criteria": {"0": "a"}}, "score criteria"),
    ({"type": "noul", "instructions": "x", "criteria": ["yes", "no"]}, "noul criteria"),
    ("not a dict", "must be an object"),
])
def test_invalid_questions(q, msg):
    with pytest.raises(QuestionError, match=msg):
        parse_question(q)


def test_parse_questions_names_the_question():
    with pytest.raises(QuestionError, match="question 'bad'"):
        parse_questions({"ok": {"type": "noul", "instructions": "x"}, "bad": {"type": "nope", "instructions": "x"}})
    with pytest.raises(QuestionError, match="non-empty object"):
        parse_questions({})
    out = parse_questions({"q1": {"type": "noul", "instructions": "x"}, "q2": {"type": "noul", "instructions": "y"}})
    assert list(out) == ["q1", "q2"]


def test_falsy_descriptions_count_as_missing():
    # the evaluated encoder renders a falsy description (0, False, {}, []) as "no description" (`text or None`)
    assert describe(0) is None and describe(0.0) is None and describe(False) is None
    assert describe({}) is None and describe([]) is None
    assert describe("0") == "0"
    d = parse_question({"type": "score", "instructions": "Rate", "criteria": [0, 1, 2]})
    assert d.texts == (None, "1", "2")
