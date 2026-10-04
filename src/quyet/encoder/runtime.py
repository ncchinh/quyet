"""EncoderModel: quyet_config.json + model.safetensors + tokenizer files -> Jev-format answers.
CUDA runs under bfloat16 autocast (as evaluated); CPU runs in float32."""
from __future__ import annotations

from pathlib import Path

from ..base import QuyetModel, Result, check_state, truncation_warning
from ..calibration import softmax_t, temperature
from ..errors import QuestionError
from ..questions import parse_questions
from . import inputs as I
from .modeling import build_model


class EncoderModel(QuyetModel):
    kind = "encoder"

    def __init__(self, path, config, device):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        self.path, self.config, self.device = Path(path), config, str(device)
        self.name = config["name"]
        self.max_len = int(config["max_len"])
        self.temps = dict(config.get("temperatures") or {})
        self.max_batch_tokens, self.max_batch_rows = 32768, 256
        model = build_model(config)
        model.load_state_dict(load_file(str(self.path / "model.safetensors")), strict=True)
        self.model = model.to(self.device).eval()
        self.tokens = I.Tokens(AutoTokenizer.from_pretrained(str(self.path)))
        self.pad_id = self.tokens.tok.pad_token_id

    def _logits(self, items):
        import torch
        out = [None] * len(items)
        cuda = self.device.startswith("cuda")
        for bi in I.length_batches([len(it["ids"]) for it in items], self.max_batch_tokens, self.max_batch_rows):
            arrays = I.collate([items[i] for i in bi], self.pad_id)
            b = {k: torch.from_numpy(v).to(self.device) for k, v in arrays.items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                lg = self.model(b).float().cpu().numpy()
            for j, i in enumerate(bi):
                out[i] = lg[j, :len(items[i]["markers"])]
        return out

    def _run(self, requests, strict):
        prepared = []
        for r in requests:
            state = r["state"]
            check_state(state)
            decisions = parse_questions(r["questions"])
            state_ids = self.tokens.encode(I.serialize_state(state))
            items = {qid: I.build_item(self.tokens, d, state_ids, isinstance(state, list), self.max_len)
                     for qid, d in decisions.items()}
            truncated = {qid: it["kept_state_tokens"] < it["state_tokens"] for qid, it in items.items()}
            warnings = []
            if any(truncated.values()):
                if strict:
                    raise QuestionError(f"state has {len(state_ids)} tokens and does not fit the {self.max_len}-token "
                                        f"window (strict)")
                warnings.append(truncation_warning(state, len(state_ids),
                                                   min(it["kept_state_tokens"] for it in items.values())))
            prepared.append((decisions, items, truncated, warnings))
        logits = iter(self._logits([it for _, items, _, _ in prepared for it in items.values()]))
        results = []
        for decisions, items, truncated, warnings in prepared:
            probs = {}
            for qid, d in decisions.items():
                lg = next(logits)
                canon = softmax_t(lg, temperature(self.temps, d.type, len(lg)))
                p = [0.0] * len(canon)
                for c, x in enumerate(canon):
                    p[items[qid]["to_label"][c]] = x
                probs[qid] = p
            results.append(Result(decisions, probs, truncated, sum(len(it["ids"]) for it in items.values()), warnings))
        return results
