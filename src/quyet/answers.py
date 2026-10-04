"""Probabilities -> Jev-format answers (the shape of TypeSafe's /v1/systemone)."""
from __future__ import annotations


def _r(x):
    return round(float(x), 4)


def format_answer(d, probs, truncated=False):
    """probs: floats aligned with d.labels, summing to 1."""
    if d.type == "choice":
        i = max(range(len(probs)), key=probs.__getitem__)
        a = {"type": "choice", "choice": d.labels[i], "confidence": _r(probs[i]),
             "probabilities": {k: _r(p) for k, p in zip(d.labels, probs)}}
    elif d.type == "score":
        a = {"type": "score", "score": _r(sum(i * p for i, p in enumerate(probs))), "confidence": _r(max(probs)),
             "legend": {k: (t if t is not None else k) for k, t in zip(d.labels, d.texts)},
             "probabilities": {k: _r(p) for k, p in zip(d.labels, probs)}}
    else:
        a = {"type": "noul", "noul": _r(probs[0]), "confidence": _r(max(probs[0], 1 - probs[0]))}
    if truncated:
        a["truncated"] = True
    return a


def build_response(model_name, decisions, probs, truncated, input_tokens, warnings):
    """decisions / probs / truncated: dicts keyed by question id, in request order."""
    return {"model": model_name,
            "answers": {qid: format_answer(d, probs[qid], truncated[qid]) for qid, d in decisions.items()},
            "usage": {"input_tokens": int(input_tokens), "output_tokens": 0},
            "warnings": list(warnings)}
