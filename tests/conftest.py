import json
import shutil

import pytest

MODERNBERT = ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8")
QWEN = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


@pytest.fixture(scope="session")
def modernbert_dir():
    from huggingface_hub import snapshot_download
    return snapshot_download(MODERNBERT[0], revision=MODERNBERT[1], allow_patterns=["config.json", *TOKENIZER_FILES])


@pytest.fixture(scope="session")
def modernbert_tok(modernbert_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(modernbert_dir)


def tiny_encoder_config(modernbert_dir, head, max_len=8192, temperatures=None):
    """A 2-layer, 64-wide ModernBERT config on the real ModernBERT-base tokenizer (random weights)."""
    from transformers import AutoConfig
    enc = AutoConfig.from_pretrained(modernbert_dir).to_dict()
    enc.update(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=2)
    if enc.get("layer_types"):
        enc["layer_types"] = enc["layer_types"][:2]
    enc.pop("_name_or_path", None)
    return {"quyet_format": 1, "name": f"tiny-{head}", "version": "0.0.0", "kind": "encoder", "languages": ["en"],
            "head": head, "max_len": max_len, "temperatures": temperatures or {}, "encoder_config": enc,
            "base_model": {"repo": MODERNBERT[0], "revision": MODERNBERT[1], "layers_kept": None}, "source": {}}


def write_encoder_export(path, modernbert_dir, head, **kw):
    import torch
    from safetensors.torch import save_file
    from quyet.encoder.modeling import build_model
    cfg = tiny_encoder_config(modernbert_dir, head, **kw)
    torch.manual_seed(0)
    model = build_model(cfg)
    path.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path / "model.safetensors"))
    for f in TOKENIZER_FILES:
        shutil.copyfile(f"{modernbert_dir}/{f}", path / f)
    (path / "quyet_config.json").write_text(json.dumps(cfg))
    return path


@pytest.fixture(scope="session")
def qwen_tok_dir():
    from huggingface_hub import snapshot_download
    return snapshot_download(QWEN[0], revision=QWEN[1], allow_patterns=[
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
        "chat_template.jinja"])


def write_llm_export(path, qwen_tok_dir, temperatures=None, prompt_version=1):
    """A 2-layer, 32-wide Qwen2 causal LM on the real Qwen3.5 tokenizer (random weights)."""
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained(qwen_tok_dir)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                         max_position_embeddings=32768, tie_word_embeddings=True))
    model.save_pretrained(path)
    tok.save_pretrained(path)
    cfg = {"quyet_format": 1, "name": "tiny-llm", "version": "0.0.0", "kind": "llm", "languages": ["en"],
           "prompt_version": prompt_version, "temperatures": temperatures or {},
           "limits": {"max_state_tokens": 6000, "max_prompt_tokens": 8000, "min_state_tokens": 256},
           "base_model": {"repo": QWEN[0], "revision": QWEN[1], "layers_kept": None}, "source": {}}
    (path / "quyet_config.json").write_text(json.dumps(cfg))
    return path
