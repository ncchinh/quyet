"""LLMModel: a merged causal LM + quyet_config.json -> Jev-format answers via one forward pass per question."""
from __future__ import annotations

from pathlib import Path

from ..base import QuyetModel, Result, check_state, truncation_warning
from ..calibration import softmax_t, temperature
from ..errors import QuestionError
from ..questions import parse_questions
from . import prompt as P


class LLMModel(QuyetModel):
    kind = "llm"

    def __init__(self, path, config, device, dtype=None, device_map=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        self.path, self.config = Path(path), config
        self.name = config["name"]
        self.tok = AutoTokenizer.from_pretrained(str(self.path))
        dtype = dtype or torch.bfloat16
        kw = dict(dtype=dtype) if device_map is None else dict(dtype=dtype, device_map=device_map)
        try:
            model = AutoModelForCausalLM.from_pretrained(str(self.path), **kw)
        except (ValueError, KeyError):
            model = AutoModelForImageTextToText.from_pretrained(str(self.path), **kw)
        if device_map is None:
            model = model.to(device)
        self.model = model.eval()
        self.device = str(self.model.get_input_embeddings().weight.device)
        self.letter_ids = P.letter_ids(self.tok)
        self.temps = dict(config.get("temperatures") or {})
        self.prompt_version = P.check_version(int(config.get("prompt_version", 1)))
        lim = config["limits"]
        self.max_state_tokens = int(lim["max_state_tokens"])
        self.max_prompt_tokens = int(lim["max_prompt_tokens"])
        self.min_state_tokens = int(lim["min_state_tokens"])
        self.max_batch_tokens = 8000
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def _prompt_ids(self, state, decisions, strict):
        """Token ids per question; the state budget shrinks until the longest prompt fits max_prompt_tokens."""
        cap = self.max_state_tokens
        for _ in range(4):
            state_text, n_state, truncated = P.truncate_state(self.tok, state, cap, compact=self.prompt_version == 2)
            ids = [self.tok(P.build_prompt(self.tok, state_text, d, self.prompt_version), add_special_tokens=False)["input_ids"]
                   for d in decisions.values()]
            over = max(len(x) for x in ids) - self.max_prompt_tokens
            if over <= 0:
                break
            cap = min(cap, n_state) - over - 16
            if cap < self.min_state_tokens:
                raise QuestionError(f"question and option text leave no room for the state: a prompt would exceed "
                                    f"{self.max_prompt_tokens} tokens")
        else:
            raise QuestionError(f"prompt exceeds {self.max_prompt_tokens} tokens")
        warnings = []
        if truncated:
            if strict:
                raise QuestionError(f"state has {n_state} tokens, over the limit of {cap} (strict)")
            warnings.append(truncation_warning(state, n_state, cap))
        return ids, truncated, warnings

    def _letter_logits(self, seqs):
        """seqs: [(ids, k)] -> letter logits over the first k letters; left-padded length-sorted batches."""
        import torch
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
        out = [None] * len(seqs)
        lid = torch.tensor(self.letter_ids, device=self.device)
        i = 0
        while i < len(order):
            j, longest = i, 0
            while j < len(order) and max(longest, len(seqs[order[j]][0])) * (j - i + 1) <= self.max_batch_tokens:
                longest = max(longest, len(seqs[order[j]][0]))
                j += 1
            j = max(j, i + 1)
            batch = order[i:j]
            L = max(len(seqs[b][0]) for b in batch)
            ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for r, b in enumerate(batch):
                s = seqs[b][0]
                ids[r, L - len(s):] = torch.tensor(s)
                att[r, L - len(s):] = 1
            with torch.no_grad():
                logits = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                                    logits_to_keep=1).logits[:, -1, :]
            sel = logits.float()[:, lid]
            for r, b in enumerate(batch):
                out[b] = sel[r, :seqs[b][1]].cpu().numpy()
            i = j
        return out

    def _run(self, requests, strict):
        prepared = []
        for r in requests:
            check_state(r["state"])
            decisions = parse_questions(r["questions"])
            ids, truncated, warnings = self._prompt_ids(r["state"], decisions, strict)
            prepared.append((decisions, ids, truncated, warnings))
        seqs = [(x, len(d.labels)) for decisions, ids, _, _ in prepared for d, x in zip(decisions.values(), ids)]
        logits = iter(self._letter_logits(seqs))
        results = []
        for decisions, ids, truncated, warnings in prepared:
            probs = {qid: softmax_t(next(logits), temperature(self.temps, d.type, len(d.labels)))
                     for qid, d in decisions.items()}
            results.append(Result(decisions, probs, {qid: truncated for qid in decisions}, sum(len(x) for x in ids),
                                  warnings))
        return results
