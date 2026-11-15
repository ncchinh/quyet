"""Inference readouts and right-padded batching from the evaluated Large path."""
import torch
from torch import nn
from torch.nn import functional as F
from .validation import check as require

class DecisionModel(nn.Module):
    def __init__(self, backbone, *, native_chunk_tokens=128):
        super().__init__()
        self.backbone = backbone
        base = self.base()
        self.option_head = nn.Linear(base.lm_head.weight.shape[1], 1, dtype=torch.float32)
        nn.init.zeros_(self.option_head.weight)
        nn.init.zeros_(self.option_head.bias)
        self.native_chunk_tokens = native_chunk_tokens

    def base(self):
        return self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone

    @property
    def device(self):
        return next(self.parameters()).device

    def _hidden(self, ids):
        base = self.base()
        options = {}
        mask = torch.ones_like(ids)
        return base.model(input_ids=ids, attention_mask=mask,
                          position_ids=torch.arange(ids.shape[1], device=ids.device)[None],
                          use_cache=False, **options).last_hidden_state

    def forward(self, request, *, compute_loss=False):
        ids = torch.tensor([request.input_ids], dtype=torch.long, device=self.device)
        hidden = self._hidden(ids)[0]
        return self._readouts(hidden, request, compute_loss=compute_loss)

    def forward_batch(self, requests, *, compute_loss=False):
        require(bool(requests), "Empty microbatch")
        if len(requests) == 1:
            return [self(requests[0], compute_loss=compute_loss)]
        lengths = [len(request.input_ids) for request in requests]
        width = max(lengths)
        base = self.base()
        cfg = getattr(base.config, "text_config", base.config)
        require(getattr(cfg, "use_bidirectional_attention", None) != "all",
                "Right-padding optimization requires causal text attention")
        pad = getattr(cfg, "pad_token_id", 0) or 0
        ids = torch.tensor([r.input_ids+[pad]*(width-n) for r, n in zip(requests, lengths)],
                           dtype=torch.long, device=self.device)
        # Each example occupies its own batch dimension. Real tokens form a
        # complete causal prefix: right padding lies strictly in their future.
        # It cannot affect any supervised position or its gradient. Keeping an
        # all-valid mask permits the causal attention kernel; padding hidden
        # states are discarded before every readout, including native EOS loss.
        hidden = self._hidden(ids)
        return [self._readouts(hidden[i, :n], request, compute_loss=compute_loss)
                for i, (request, n) in enumerate(zip(requests, lengths))]

    def _readouts(self, hidden, request, *, compute_loss=False):
        require(not compute_loss, "This runtime is inference-only")
        base = self.base()
        losses, logits = [], []
        for d in request.decisions:
            positions = torch.tensor(d["positions"], device=hidden.device)
            if d["readout"] == "head":
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    z = self.option_head(hidden[positions].float().to(self.option_head.weight.device)).squeeze(-1)
            else:
                symbols = torch.tensor(d["symbol_ids"], device=base.lm_head.weight.device)
                # Matches the accepted alphabet readout: no full-vocabulary
                # allocation and no extra logit softcap on classifier logits.
                z = F.linear(hidden[positions[0]].to(base.lm_head.weight.device), base.lm_head.weight[symbols]).float()
            require(len(z) == len(d["option_ids"]), "Option/readout width differs")
            logits.append(z)
        result = torch.stack(losses) if losses else hidden.new_empty((0,), dtype=torch.float32)
        require(bool(torch.isfinite(result).all()), "Nonfinite decision objective")
        return result, logits

    def prefix_cache_decoder(self):
        return self.base().model

    def prefix_cache_readout(self, hidden, decision):
        from .request import Request
        return self._readouts(hidden, Request([], [decision]), compute_loss=False)[1][0]

