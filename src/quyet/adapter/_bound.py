"""Evaluated W8 renderer; only construction is relocated to the model binding."""
import json
import numpy as np

DELIM_TOKEN = "<unused0>"


OPTION_CEILING = 256


MODES = ("letter", "per_option_head", "alphabet256")


TYPES = ["choice", "score", "noul"]


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


KIND = {"choice": "Choose the option that fits best.",
        "score": "Choose the level that fits best (levels are ordered from lowest to highest).",
        "noul": "Choose A if the statement is true for this state, B if it is not."}


def option_texts_and_ids(d):
    """(option texts, option ids, gold index) in the supplied order; same semantics as eval_llm.options_of."""
    t, opts = d["type"], d["options"]
    if t == "choice":
        ids = [str(o["id"]) for o in opts]
        if len(set(ids)) != len(ids):
            raise ValueError("choice option ids must be unique")
        return [o.get("text") or str(o["id"]) for o in opts], ids, ids.index(str(d["target"]))
    if t == "score":
        levels = sorted(opts, key=lambda o: int(o["id"]))
        return [o.get("text") or str(o["id"]) for o in levels], [str(o["id"]) for o in levels], int(d["target"])
    if t == "noul":
        by = {str(o["id"]).lower(): o.get("text") for o in opts}
        return ([by.get("true") or "true", by.get("false") or "false"], ["true", "false"],
                0 if str(d["target"]).lower() == "true" else 1)
    raise ValueError(f"unknown decision type {t!r}")


def serialize_state(state):
    """diet_d: strings as they are, structured states as compact JSON."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))


class Renderer:
    """Tokenizer-resolved rendering of one decision. render() returns an example dict or an UNSUPPORTED reason."""

    def __init__(self, tok, binding, window, reference=False):
        self.tok, self.window = tok, window
        self.mode = 'alphabet256' if reference else binding['mode']
        self.prefix, self.suffix = binding['prefix_ids'], binding['suffix_ids']
        self.delim_id = binding['delimiter_id']
        self.symbols = binding['reference_symbols'] if reference else binding['symbols']
        self.symbol_ids = binding['reference_symbol_ids'] if reference else binding['symbol_ids']

    def encode(self, text):
        """User text: control tokens are escaped (split into ordinary pieces), nothing is added."""
        return self.tok(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]

    def user_text(self, state, d):
        return f"State:\n{serialize_state(state)}\n\nQuestion: {d['question']}\n{KIND[d['type']]}\n\nOptions:"

    def render(self, state, d, rng=None):
        if d.get("target") in (None, ""):
            return "no_target"
        try:
            texts, option_ids, gold = option_texts_and_ids(d)
        except (ValueError, KeyError, TypeError, AttributeError):
            return "unparseable"
        k = len(texts)
        if k < 2:
            return "single_option"
        if k > OPTION_CEILING:
            return "over_ceiling"
        head = self.mode == "per_option_head" and d["type"] == "choice"
        if not head and k > len(self.symbols):
            return "over_ceiling" if self.mode == "alphabet256" else "over_letters"
        order = list(range(k))
        if rng is not None and d["type"] == "choice":
            rng.shuffle(order)
        texts, option_ids, gold = [texts[i] for i in order], [option_ids[i] for i in order], order.index(gold)
        ids = list(self.prefix) + self.encode(self.user_text(state, d))
        pos = []
        if head:
            for t in texts:
                ids.extend(self.encode("\n" + t))
                ids.append(self.delim_id)
                pos.append(len(ids) - 1)
        else:
            for sym, t in zip(self.symbols, texts):
                ids.extend(self.encode(f"\n{sym}. {t}"))
            ids.extend(self.suffix)
        if len(ids) > self.window:
            return "over_window"
        ex = {"ids": np.asarray(ids, dtype=np.int32), "k": k, "gold": gold, "type": TYPES.index(d["type"]),
              "readout": "head" if head else "letter", "option_ids": option_ids, "order": order,
              "target_id": option_ids[gold]}
        if head:
            ex["pos"] = np.asarray(pos, dtype=np.int32)
            if int((ex["ids"] == self.delim_id).sum()) != k:
                raise RuntimeError("delimiter token produced from user text; rendering is ambiguous")
        return ex

