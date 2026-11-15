"""Model-bound W8 rendering, with the evaluated public-question conversion."""
import json
from tokenizers import Tokenizer
from ._bound import Renderer as BoundRenderer


class Renderer:
    def __init__(self, path, config):
        self.cfg = config['renderer']
        self.backend = Tokenizer.from_file(str(path / 'tokenizer.json'))
        self.backend.no_truncation()
        self.backend.no_padding()
        backend = self.backend

        class Backend:
            def __call__(self, text, add_special_tokens=False, split_special_tokens=True):
                backend.encode_special_tokens = split_special_tokens
                return dict(input_ids=backend.encode(text, add_special_tokens=add_special_tokens).ids)

        self.head = BoundRenderer(Backend(), self.cfg, config['limits']['max_input_tokens'])
        self.alpha = BoundRenderer(Backend(), self.cfg, config['limits']['max_input_tokens'], reference=True)

    # -- build_packet.request, verbatim semantics ------------------------------------------------------------------
    @staticmethod
    def convert_question(q, key):
        kind = q["type"]
        question = q.get("instructions") or key
        if not isinstance(question, str):
            question = json.dumps(question, ensure_ascii=False, separators=(",", ":"))
        if kind == "choice":
            criteria = q["criteria"]
            pairs = criteria.items() if isinstance(criteria, dict) else ((k, k) for k in criteria)
            opts = [{"id": str(k), "text": (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))) if v else str(k)}
                    for k, v in pairs]
            return {"type": kind, "question": question, "options": opts, "target": opts[0]["id"]}
        if kind == "noul":
            return {"type": kind, "question": question,
                    "options": [{"id": "true", "text": "true"}, {"id": "false", "text": "false"}], "target": "true"}
        if kind == "score":
            assert isinstance(q["criteria"], list)
            return {"type": kind, "question": question,
                    "options": [{"id": str(i), "text": (text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, separators=(",", ":"))) if text else None}
                                for i, text in enumerate(q["criteria"])], "target": 0}
        raise ValueError("Unbound public benchmark format: " + str(kind))

    def _pack(self, ex, kind, key, symbol_ids):
        return {"question_id": key, "input_ids": ex["ids"].tolist(), "decision": {
            "readout": ex["readout"], "decision_type": kind, "option_ids": ex["option_ids"],
            "positions": ex["pos"].tolist() if ex["readout"] == "head" else [len(ex["ids"]) - 1],
            "symbol_ids": symbol_ids[:ex["k"]], "loss_weight": 1.0}}

    def request_public(self, state, question, key):
        return self.request_v3(state, self.convert_question(question, key), key)

    def request_jev(self, state, q, key):
        d = self.convert_question(q, key)
        ex = self.head.render(state, d)
        if isinstance(ex, str):
            return {"question_id": key, "unsupported": ex}
        return self._pack(ex, d["type"], key, self.cfg["symbol_ids"])

    # -- V3/V5-format decisions, store profile rule ------------------------------------------------------------------
    @staticmethod
    def placeholder(d):
        """Visible fields only; the target becomes a fixed placeholder (first option / level 0 / true)."""
        out = {"type": d["type"], "question": d["question"],
               "options": [{"id": o["id"], **({"text": o["text"]} if "text" in o else {})} for o in d["options"]]}
        if d["type"] == "choice":
            out["target"] = str(d["options"][0]["id"])
        elif d["type"] == "score":
            out["target"] = 0
        else:
            out["target"] = "true"
        return out

    def render_v3(self, state, d):
        """(example or unsupported reason, profile) under the store's profile rule."""
        vis = self.placeholder(d)
        if d["type"] == "choice":
            return self.head.render(state, vis), "w8.per_option_head"
        return self.alpha.render(state, vis), "w8-gemma4-alphabet256-diet_d-32k-reference-v1"

    def request_v3(self, state, d, key):
        ex, profile = self.render_v3(state, d)
        if isinstance(ex, str):
            return {"question_id": key, "unsupported": ex, "profile_id": profile}
        sym = self.cfg["symbol_ids"] if profile == "w8.per_option_head" else self.cfg["reference_symbol_ids"]
        r = self._pack(ex, d["type"], key, sym)
        r["profile_id"] = profile
        return r
