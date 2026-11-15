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
    from huggingface_hub.constants import HF_HUB_OFFLINE
    return Path(snapshot_download(str(name_or_path), revision=revision, local_files_only=HF_HUB_OFFLINE))


def read_config(path):
    f = Path(path) / "quyet_config.json"
    if not f.is_file():
        raise ValueError(f"{path} is not a Quyet model: quyet_config.json is missing")
    cfg = json.loads(f.read_text(encoding="utf8"))
    if cfg.get("quyet_format") == 2:
        if cfg.get("kind") not in ("adapter", "student", "merged"):
            raise ValueError("quyet_format 2 requires kind 'adapter', 'student' or 'merged'")
        return cfg
    if cfg.get("quyet_format") != FORMAT:
        raise ValueError(f"quyet_format {cfg.get('quyet_format')!r} is not supported by this version of the package "
                         f"(reads format {FORMAT}); upgrade quyet")
    if cfg.get("kind") not in KINDS:
        raise ValueError(f"unknown model kind {cfg.get('kind')!r}; expected one of {list(KINDS)}")
    return cfg


def load(name_or_path, *, device=None, revision=None, dtype=None, device_map=None, base=None, prefix_cache=False, prefix_cache_max_bytes=8 * 1024 ** 3):
    """Load a Quyet model. device: "cuda" when available, else "cpu". dtype (default bfloat16) and device_map
    (e.g. "auto" for multiple GPUs; requires quyet[multi-gpu]) apply to LLM and adapter models.
    base applies only to format-2 adapter models; prefix_cache also supports format-2 students."""
    path = resolve(name_or_path, revision)
    cfg = read_config(path)
    if cfg["kind"] == "merged":
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        from .merged.runtime import MergedModel
        return MergedModel(path, cfg, device, dtype=dtype, device_map=device_map, base=base, prefix_cache=prefix_cache, prefix_cache_max_bytes=prefix_cache_max_bytes)
    if cfg["kind"] == "student":
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        from .student.runtime import StudentModel
        return StudentModel(path, cfg, device, dtype=dtype, device_map=device_map, base=base, prefix_cache=prefix_cache, prefix_cache_max_bytes=prefix_cache_max_bytes)
    if cfg["kind"] == "adapter":
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        from .adapter.runtime import AdapterModel
        return AdapterModel(path, cfg, device, dtype=dtype, device_map=device_map, base=base, prefix_cache=prefix_cache, prefix_cache_max_bytes=prefix_cache_max_bytes)
    if base is not None or prefix_cache or prefix_cache_max_bytes != 8 * 1024 ** 3:
        raise ValueError("base and prefix_cache apply to format-2 adapter models only")
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
