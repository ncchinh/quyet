"""Find a Quyet model (local directory or Hugging Face repo id) and load it."""
from __future__ import annotations

import json
from pathlib import Path

FORMAT = 1
KINDS = ("encoder", "llm")


def resolve(name_or_path, revision=None):
    p = Path(str(name_or_path)).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(name_or_path), revision=revision))


def read_config(path):
    f = Path(path) / "quyet_config.json"
    if not f.is_file():
        raise ValueError(f"{path} is not a Quyet model: quyet_config.json is missing")
    cfg = json.loads(f.read_text(encoding="utf8"))
    if cfg.get("quyet_format") != FORMAT:
        raise ValueError(f"quyet_format {cfg.get('quyet_format')!r} is not supported by this version of the package "
                         f"(reads format {FORMAT}); upgrade quyet")
    if cfg.get("kind") not in KINDS:
        raise ValueError(f"unknown model kind {cfg.get('kind')!r}; expected one of {list(KINDS)}")
    return cfg


def load(name_or_path, *, device=None, revision=None, dtype=None, device_map=None):
    """Load a Quyet model. device: "cuda" when available, else "cpu". dtype (default bfloat16) and device_map
    (e.g. "auto" to spread Quyet-1.0-Large over several GPUs; needs `pip install quyet[multi-gpu]`) apply to LLMs only."""
    path = resolve(name_or_path, revision)
    cfg = read_config(path)
    if cfg["kind"] == "encoder":
        if device_map is not None:
            raise ValueError("device_map applies to LLM models only; use device= for encoders")
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        from .encoder.runtime import EncoderModel
        return EncoderModel(path, cfg, device)
    if device is None and device_map is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from .llm.runtime import LLMModel
    return LLMModel(path, cfg, device, dtype=dtype, device_map=device_map)
