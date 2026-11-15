"""Jev /v1/systemone questions -> validated Decision objects shared by both runtimes.

A question is {"type": "choice" | "score" | "noul", "instructions": str, "criteria": ...}:
  choice  criteria {label: description} (or a list of labels)  -> answer keys = the labels, in the given order
  score   criteria [level description, ...] lowest first         -> answer keys "0".."K-1"
  noul    criteria {"true": description, "false": description}  -> answer keys ("true", "false"); both optional
Descriptions keep None where none was given; each runtime applies the fallback text it was trained with.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .errors import QuestionError

TYPES = ("choice", "score", "noul")
MAX_OPTIONS = 10
DEFAULT_MAX_OPTIONS = {"choice": 10, "score": 10, "noul": 2}


@dataclass(frozen=True)
class Decision:
    type: str
    instructions: str
    labels: tuple
    texts: tuple


def describe(value):
    """Option description as text: strings as-is, other JSON values as compact JSON. Falsy values (None, "", 0,
    False, {}, []) mean "no description", as in the evaluated encoder path (`text or None`)."""
    if not value:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def parse_question(q, max_options=None) -> Decision:
    if not isinstance(q, dict):
        raise QuestionError("a question must be an object with type, instructions and criteria")
    t, instr, crit = q.get("type"), q.get("instructions"), q.get("criteria")
    if t not in TYPES:
        raise QuestionError(f"unknown question type {t!r}; expected one of {list(TYPES)}")
    if not isinstance(instr, str) or not instr.strip():
        raise QuestionError("instructions must be a non-empty string")
    if t == "choice":
        if isinstance(crit, dict):
            pairs = [(str(k), describe(v)) for k, v in crit.items()]
        elif isinstance(crit, list):
            pairs = [(str(k), None) for k in crit]
        else:
            raise QuestionError("choice criteria must be an object {label: description} or a list of labels")
        if len({k for k, _ in pairs}) != len(pairs):
            raise QuestionError("choice labels must be unique")
    elif t == "score":
        if not isinstance(crit, list):
            raise QuestionError("score criteria must be a list of level descriptions, lowest first")
        pairs = [(str(i), describe(c)) for i, c in enumerate(crit)]
    else:
        crit = {} if crit is None else crit
        if not isinstance(crit, dict):
            raise QuestionError("noul criteria must be an object with optional keys true and false")
        pairs = [("true", describe(crit.get("true"))), ("false", describe(crit.get("false")))]
    if len(pairs) < 2:
        raise QuestionError(f"{t} needs at least 2 options")
    ceiling = (max_options or DEFAULT_MAX_OPTIONS)[t]
    if len(pairs) > ceiling:
        raise QuestionError(f"{t} supports at most {ceiling} options, got {len(pairs)}")
    return Decision(t, instr, tuple(k for k, _ in pairs), tuple(v for _, v in pairs))


def parse_questions(questions, max_options=None) -> dict:
    if not isinstance(questions, dict) or not questions:
        raise QuestionError("questions must be a non-empty object {question_id: question}")
    out = {}
    for qid, q in questions.items():
        try:
            out[str(qid)] = parse_question(q, max_options)
        except QuestionError as e:
            raise QuestionError(f"question {qid!r}: {e}") from None
    return out


def option_limits(config, *, choice_ceiling=256):
    """Model-declared limits; legacy models retain their original option envelope."""
    limits = dict(DEFAULT_MAX_OPTIONS)
    declared = config.get("limits", {}).get("max_options", {})
    if not isinstance(declared, dict) or set(declared) - set(limits):
        raise ValueError("limits.max_options must map choice, score and noul to integers")
    limits.update(declared)
    for kind, ceiling in (("choice", choice_ceiling), ("score", 10), ("noul", 2)):
        value = limits[kind]
        if type(value) is not int or not 2 <= value <= ceiling:
            raise ValueError(f"limits.max_options.{kind} must be an integer in 2..{ceiling}")
    return limits
