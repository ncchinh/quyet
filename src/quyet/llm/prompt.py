"""Prompt for the LLM letter readout, exactly as the Quyet LLMs were trained and evaluated.

The prompt holds the state, the question and the options lettered A..J. Two prompt versions (quyet_config.json
"prompt_version"): 1 = the training prompt (Quyet-1.0-Medium); 2 = the "diet" prompt for Quyet-1.0-Large: no system
message, no closing line, compact JSON states, the trained per-type instructions kept. Version 2 cuts ~68 input tokens
per decision; on held-out data its accuracy is about 0.3 pt below version 1. Its temperatures were refit for it. The next-token distribution over those
letters (renormalised, temperature-scaled) is the decision distribution. Nothing is generated.
  choice: options in the given order; score: levels 0..k-1 in order; noul: A = true description, B = false description.
"""
from __future__ import annotations

import json

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_LETTERS = 10
SYSTEM = ("You are a decision model. Read the state and the question, then choose exactly one option. "
          "Base the decision only on the state and the option descriptions. Reply with the option letter only.")
KIND = {"choice": "Choose the option that fits best.",
        "score": "Choose the level that fits best (levels are ordered from lowest to highest).",
        "noul": "Choose A if the statement is true for this state, B if it is not."}


def option_texts(d):
    """Option texts in letter order, with the fallbacks the models were trained with."""
    if d.type == "noul":
        return [d.texts[0] or "true", d.texts[1] or "false"]
    if d.type == "score":
        return [t if t is not None else str(i) for i, t in enumerate(d.texts)]
    return [t if t is not None else k for k, t in zip(d.labels, d.texts)]


PROMPT_VERSIONS = (1, 2)
CLOSING = "Answer with one letter."


def check_version(version):
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt_version {version!r}; this package knows {list(PROMPT_VERSIONS)}")
    return version


def serialize_state(state, compact=False):
    if isinstance(state, str):
        return state
    if compact:
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(state, ensure_ascii=False, indent=1)


def truncate_state(tok, state, max_tokens, compact=False):
    """(state text for the prompt, state tokens before truncation, truncated?). Lists keep the tail, others the head.
    compact: JSON without indentation (prompt version 2)."""
    s = serialize_state(state, compact)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return s, len(ids), False
    is_list = isinstance(state, list)
    keep = ids[-max_tokens:] if is_list else ids[:max_tokens]
    return ("… " if is_list else "") + tok.decode(keep) + ("" if is_list else " …"), len(ids), True


def build_prompt(tok, state_text, d, version=1):
    check_version(version)
    opts = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(option_texts(d)))
    user = f"State:\n{state_text}\n\nQuestion: {d.instructions}\n{KIND[d.type]}\n\nOptions:\n{opts}"
    if version == 1:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{user}\n\n{CLOSING}"}]
    else:
        msgs = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def letter_ids(tok, n=MAX_LETTERS):
    ids = []
    for letter in LETTERS[:n]:
        enc = tok.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"letter {letter} is not a single token: {enc}")
        ids.append(enc[0])
    return ids
