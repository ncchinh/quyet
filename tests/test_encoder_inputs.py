import numpy as np
import pytest

from quyet.encoder import inputs as I
from quyet.errors import QuestionError
from quyet.questions import parse_question


def toks(tok):
    return I.Tokens(tok)


def build(tok, q, state="the state", max_len=8192):
    t = toks(tok)
    d = parse_question(q)
    return t, d, I.build_item(t, d, t.encode(I.serialize_state(state)), isinstance(state, list), max_len)


def test_layout(modernbert_tok):
    tok = modernbert_tok
    _, _, it = build(tok, {"type": "choice", "instructions": "Pick one", "criteria": {"a": "apple", "b": None}})
    ids = it["ids"]
    assert ids[0] == tok.cls_token_id and ids[it["q_end"]] == tok.sep_token_id and ids[-1] == tok.sep_token_id
    assert tok.decode(ids[1:it["q_end"]]) == "choice question: Pick one"
    assert [ids[m] for m in it["markers"]] == [tok.mask_token_id] * 2
    (s0, e0), (s1, e1) = it["spans"]
    assert tok.decode(ids[it["markers"][0] + 1:s0]) == " a:" and tok.decode(ids[s0:e0]) == " apple"
    assert tok.decode(ids[it["markers"][1] + 1:s1]) == " b" and s1 == e1          # no text: label only, empty span
    assert ids[e1] == tok.sep_token_id and it["st_start"] == e1 + 1
    assert tok.decode(ids[it["st_start"]:it["st_end"]]) == "the state"
    assert it["qtype"] == 0 and it["to_label"] == [0, 1]


def test_score_and_noul_fallback_texts(modernbert_tok):
    tok = modernbert_tok
    _, _, it = build(tok, {"type": "score", "instructions": "Rate", "criteria": [None, "high"]})
    (s0, e0), (s1, e1) = it["spans"]
    assert tok.decode(it["ids"][it["markers"][0] + 1:e0]) == " level 0: level 0"
    assert tok.decode(it["ids"][it["markers"][1] + 1:e1]) == " level 1: high"
    _, _, it = build(tok, {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes it is"}})
    (s0, e0), (s1, e1) = it["spans"]
    assert tok.decode(it["ids"][it["markers"][0] + 1:e0]) == " false: no, the statement does not hold"
    assert tok.decode(it["ids"][it["markers"][1] + 1:e1]) == " true: yes it is"
    assert it["to_label"] == [1, 0] and it["qtype"] == 2


def test_state_serialization():
    assert I.serialize_state("x") == "x"
    assert I.serialize_state({"a": "Hà Nội"}) == '{"a": "Hà Nội"}'
    assert I.serialize_state([{"role": "user", "content": "hi"}]) == '[{"role": "user", "content": "hi"}]'


def test_dict_state_keeps_head(modernbert_tok):
    tok = modernbert_tok
    t, d, full = build(tok, {"type": "noul", "instructions": "x"}, state="one two three four five six seven")
    head_len = full["st_start"] + 1
    _, _, it = build(tok, {"type": "noul", "instructions": "x"}, state="one two three four five six seven",
                     max_len=head_len + 2)
    assert it["kept_state_tokens"] == 2 < it["state_tokens"]
    assert tok.decode(it["ids"][it["st_start"]:it["st_end"]]) == "one two"


def test_list_state_keeps_tail(modernbert_tok):
    tok = modernbert_tok
    state = [{"role": "user", "content": "first"}, {"role": "user", "content": "last turn"}]
    t, d, full = build(tok, {"type": "noul", "instructions": "x"}, state=state)
    _, _, it = build(tok, {"type": "noul", "instructions": "x"}, state=state, max_len=full["st_start"] + 1 + 4)
    kept = tok.decode(it["ids"][it["st_start"]:it["st_end"]])
    assert it["kept_state_tokens"] == 4 and kept.endswith('"}]') and "first" not in kept


def test_unrepresentable_question_raises(modernbert_tok):
    with pytest.raises(QuestionError, match="token window"):
        build(modernbert_tok, {"type": "choice", "instructions": "word " * 50, "criteria": ["a", "b"]}, max_len=20)


def test_control_strings_stay_text(modernbert_tok):
    tok = modernbert_tok
    _, _, it = build(tok, {"type": "choice", "instructions": "[CLS] pick", "criteria": {"a": "[SEP]", "b": "[MASK]"}},
                     state="[SEP] [CLS] [MASK] [PAD]")
    special = set(tok.all_special_ids)
    structural = {0, it["q_end"], len(it["ids"]) - 1, it["st_start"] - 1, *it["markers"]}
    assert not [i for p, i in enumerate(it["ids"]) if i in special and p not in structural]


def test_collate_and_batches(modernbert_tok):
    t = toks(modernbert_tok)
    d2 = parse_question({"type": "choice", "instructions": "x", "criteria": ["a", "b"]})
    d3 = parse_question({"type": "choice", "instructions": "x", "criteria": ["a", "b", "c"]})
    a = I.build_item(t, d2, t.encode("short"), False, 8192)
    b = I.build_item(t, d3, t.encode("a much longer state text here"), False, 8192)
    batch = I.collate([a, b], pad_id=modernbert_tok.pad_token_id)
    assert batch["input_ids"].shape == (2, len(b["ids"])) and batch["marker_pos"].shape == (2, 3)
    assert batch["attention_mask"][0].sum() == len(a["ids"]) and batch["marker_mask"][0].tolist() == [True, True, False]
    assert batch["evidence_mask"][0, 0] and not batch["evidence_mask"][0, a["q_end"]]
    assert batch["qtype"].tolist() == [0, 0]
    assert [x.tolist() for x in I.length_batches([5, 1, 3], max_tokens=6, max_rows=8)] == [[1, 2], [0]]
    assert [x.tolist() for x in I.length_batches([5, 1, 3], max_tokens=100, max_rows=2)] == [[1, 2], [0]]
