"""Decision -> encoder input sequence: the exact layout the Quyet encoders were trained on.

    CLS "<type> question: <instructions>" SEP {MASK " <label>[:]" [" <text>"]}* SEP state SEP

Each option is a MASK marker, its label, then its description as a separate text span (empty when there is no
description, in which case the label has no colon). Only the state is truncated: list (conversation) states keep
their end, all others keep their beginning. Option rendering follows Convai Innovations' Laya (Apache-2.0).
"""
from __future__ import annotations

import json

import numpy as np

from ..errors import QuestionError

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def serialize_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def canonical(d):
    """(option (label, text) parts in the model's order, map from that order to the index in d.labels)."""
    if d.type == "choice":
        return list(zip(d.labels, d.texts)), list(range(len(d.labels)))
    if d.type == "score":
        return ([("level %d" % i, t if t is not None else "level %d" % i) for i, t in enumerate(d.texts)],
                list(range(len(d.labels))))
    t_true, t_false = d.texts
    return ([("false", t_false if t_false is not None else "no, the statement does not hold"),
             ("true", t_true if t_true is not None else "yes, the statement holds")], [1, 0])


class Tokens:
    """Plain-text tokenization: literal special-token strings in user text stay text."""

    def __init__(self, tok):
        self.tok = tok
        self.special = frozenset(tok.all_special_ids)

    def encode(self, text):
        ids = self.tok(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        if any(i in self.special for i in ids):
            raise QuestionError(f"text produced a special token: {text[:80]!r}")
        return ids


def build_item(tokens, d, state_ids, keep_end, max_len):
    tok = tokens.tok
    parts, to_label = canonical(d)
    head = [tok.cls_token_id] + tokens.encode("%s question: %s" % (d.type, d.instructions))
    q_end = len(head)
    head.append(tok.sep_token_id)
    markers, spans = [], []
    for label, text in parts:
        markers.append(len(head))
        head += [tok.mask_token_id] + tokens.encode(" " + label + (":" if text is not None else ""))
        s = len(head)
        if text is not None:
            head += tokens.encode(" " + text)
        spans.append((s, len(head)))
    head.append(tok.sep_token_id)
    room = max_len - len(head) - 1
    if room < 0:
        raise QuestionError(f"question and options need {len(head) + 1} tokens, over the {max_len}-token window")
    n = len(state_ids)
    kept = list(state_ids[max(0, n - room):] if keep_end else state_ids[:room])
    st_start = len(head)
    return dict(ids=head + kept + [tok.sep_token_id], markers=markers, spans=spans, q_end=q_end, st_start=st_start,
                st_end=st_start + len(kept), qtype=QTYPES[d.type], to_label=to_label, state_tokens=n,
                kept_state_tokens=len(kept))


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    k = max(len(it["markers"]) for it in items)
    ids = np.full((n, L), pad_id, np.int64)
    att = np.zeros((n, L), np.int64)
    evid = np.zeros((n, L), bool)
    mpos = np.zeros((n, k), np.int64)
    mmask = np.zeros((n, k), bool)
    s0 = np.zeros((n, k), np.int64)
    s1 = np.zeros((n, k), np.int64)
    for i, it in enumerate(items):
        t, m = it["ids"], it["markers"]
        ids[i, :len(t)] = t
        att[i, :len(t)] = 1
        evid[i, 0] = True                                     # CLS
        evid[i, 1:it["q_end"]] = True                         # question tokens
        evid[i, it["st_start"]:it["st_end"]] = True           # (possibly truncated) state tokens
        mpos[i, :len(m)] = m
        mmask[i, :len(m)] = True
        s0[i, :len(m)] = [s for s, _ in it["spans"]]
        s1[i, :len(m)] = [e for _, e in it["spans"]]
    return dict(input_ids=ids, attention_mask=att, evidence_mask=evid, marker_pos=mpos, marker_mask=mmask,
                span_start=s0, span_end=s1, qtype=np.asarray([it["qtype"] for it in items], np.int64))


def length_batches(lengths, max_tokens, max_rows):
    """Sort by length (stable); cut when padded tokens or rows would overflow. A long item may stand alone."""
    order = np.argsort(np.asarray(lengths), kind="stable")
    batches, cur, cur_max = [], [], 0
    for i in order.tolist():
        L = int(lengths[i])
        if cur and (max(cur_max, L) * (len(cur) + 1) > max_tokens or len(cur) >= max_rows):
            batches.append(np.asarray(cur, dtype=np.int64))
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(np.asarray(cur, dtype=np.int64))
    return batches
