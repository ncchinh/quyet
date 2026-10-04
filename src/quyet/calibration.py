"""Per-bucket temperature scaling. Encoder temperatures are keyed "<type>:<size>" (e.g. "choice:3-5"); LLM
temperatures are keyed by type. A missing key means 1.0 (uncalibrated)."""
from __future__ import annotations

import numpy as np


def size_bucket(k):
    return "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"


def temperature(temps, qtype, k):
    return float(temps.get(f"{qtype}:{size_bucket(k)}", temps.get(qtype, 1.0)))


def softmax_t(logits, t):
    z = np.asarray(logits, np.float64) / t
    p = np.exp(z - z.max())
    return (p / p.sum()).tolist()
