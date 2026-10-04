"""Encoder decision models: a ModernBERT-family encoder plus one option-scoring head (inference only).

Every head jointly reads question + options + state once, conditions on the question type and scores each option
with a shared scalar scorer; padded option slots are masked before the softmax.
  option_query_v1    encoder states -> Linear(d, 384) + type embedding; option query = GELU(Linear(2h, h)[marker;
                     mean(option text)]); one cross-attention block into CLS + question + state; one self-attention block
                     over the option queries; LayerNorm -> Linear -> GELU -> Linear.
  option_query_wide  the same with h = 1024, 16 heads and two (cross-attention, self-attention) pairs.
  refine_pool_v1     encoder states + type embedding -> two full-sequence pre-norm Transformer layers -> marker + mean
                     option text -> GELU(Linear(2d, d)) -> LayerNorm -> Linear -> GELU -> Linear. This head follows the
                     decision head of Convai Innovations' Laya (Apache-2.0).
"""
from __future__ import annotations

import torch
import torch.nn as nn

HEADS = ("option_query_v1", "refine_pool_v1", "option_query_wide")
MASKED_LOGIT = -1e4


def span_pool(x, marker_vec, span_start, span_end):
    """Mean of x over each option's text span [start, end); empty spans fall back to the option's marker vector."""
    pos = torch.arange(x.size(1), device=x.device)[None, None, :]
    inside = (pos >= span_start[:, :, None]) & (pos < span_end[:, :, None])
    n = inside.sum(-1, keepdim=True)
    w = inside.to(x.dtype) / n.clamp(min=1).to(x.dtype)
    return torch.where(n > 0, torch.bmm(w, x), marker_vec)


def gather_positions(x, pos):
    return torch.gather(x, 1, pos.clamp(min=0)[:, :, None].expand(-1, -1, x.size(-1)))


class CrossBlock(nn.Module):
    def __init__(self, h, heads, dropout):
        super().__init__()
        self.ln_q, self.ln_m, self.ln_f = nn.LayerNorm(h), nn.LayerNorm(h), nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, heads, dropout=dropout, batch_first=True)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(h, 4 * h), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * h, h))

    def forward(self, q, mem, mem_valid):
        m = self.ln_m(mem)
        a = self.attn(self.ln_q(q), m, m, key_padding_mask=~mem_valid, need_weights=False)[0]
        q = q + self.drop1(a)
        return q + self.drop2(self.ffn(self.ln_f(q)))


class OptionQueryHead(nn.Module):
    def __init__(self, d, h=384, heads=6, dropout=0.0, blocks=1):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.type_emb = nn.Embedding(3, h)
        self.query = nn.Sequential(nn.Linear(2 * h, h), nn.GELU())
        self.cross = CrossBlock(h, heads, dropout)
        self.refine = nn.TransformerEncoderLayer(h, heads, 4 * h, dropout, activation="gelu", batch_first=True,
                                                 norm_first=True)
        self.extra = nn.ModuleList(nn.ModuleList([
            CrossBlock(h, heads, dropout),
            nn.TransformerEncoderLayer(h, heads, 4 * h, dropout, activation="gelu", batch_first=True, norm_first=True)])
            for _ in range(blocks - 1)) if blocks > 1 else None
        self.scorer = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, hs, b):
        x = self.proj(hs) + self.type_emb(b["qtype"])[:, None, :]
        mk = gather_positions(x, b["marker_pos"])
        q = self.query(torch.cat([mk, span_pool(x, mk, b["span_start"], b["span_end"])], -1))
        pairs = [(self.cross, self.refine)] + ([tuple(p) for p in self.extra] if self.extra is not None else [])
        for cross, refine in pairs:
            q = cross(q, x, b["evidence_mask"])
            q = refine(q, src_key_padding_mask=~b["marker_mask"])
        return self.scorer(q).squeeze(-1)


class RefinePoolHead(nn.Module):
    def __init__(self, d, layers=2, dropout=0.0):
        super().__init__()
        self.type_emb = nn.Embedding(3, d)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(d, d // 64, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers))
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, hs, b):
        x = hs + self.type_emb(b["qtype"])[:, None, :]
        pad = ~b["attention_mask"].bool()
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad)
        mk = gather_positions(x, b["marker_pos"])
        return self.scorer(self.fuse(torch.cat([mk, span_pool(x, mk, b["span_start"], b["span_end"])], -1))).squeeze(-1)


def make_head(kind, d, dropout=0.0):
    if kind == "option_query_v1":
        return OptionQueryHead(d, 384, 6, dropout)
    if kind == "option_query_wide":
        return OptionQueryHead(d, 1024, 16, dropout, blocks=2)
    if kind == "refine_pool_v1":
        if d % 64:
            raise ValueError(f"refine_pool_v1 needs a width divisible by 64, got {d}")
        return RefinePoolHead(d, 2, dropout)
    raise ValueError(f"unknown head {kind!r}; expected one of {list(HEADS)}")


class DecisionModel(nn.Module):
    def __init__(self, encoder, head_kind, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.head_kind = head_kind
        self.head = make_head(head_kind, encoder.config.hidden_size, dropout)

    def forward(self, b):
        hs = self.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        return self.head(hs, b).float().masked_fill(~b["marker_mask"], MASKED_LOGIT)


def build_model(config):
    """Skeleton from quyet_config.json (random weights until a state dict is loaded)."""
    from transformers import AutoConfig, AutoModel
    enc = AutoModel.from_config(AutoConfig.for_model(**config["encoder_config"]), attn_implementation="sdpa")
    return DecisionModel(enc, config["head"])
