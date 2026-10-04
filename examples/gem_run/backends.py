"""Optional public SDK adapters. Importing this module loads no model libraries."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import platform

from .engine import ACTIONS

LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    source: str
    language: str
    revision: str | None = None
    subfolder: str | None = None

    def portable_metadata(self):
        path = Path(self.source).expanduser()
        local = path.is_dir() or path.is_absolute() or self.source.startswith(("./", "../", "~"))
        source = {"kind": "local", "name": path.name} if local else {
            "kind": "huggingface", "repo": self.source, "revision": self.revision}
        if self.subfolder:
            source["subfolder"] = self.subfolder
        labels = {"small_en": "Quyet Small-EN", "small": "Quyet Small",
                  "laya_english": "Laya EN", "laya_multilingual": "Laya Multilingual"}
        return dict(name=self.name, display_name=labels.get(self.name, path.name),
                    backend=self.backend, language=self.language, source=source)


def normalize_answer(output, language):
    """Reject malformed/truncated model output before it can steer or score."""
    try:
        answer = output["answers"]["move"]
        action = answer["choice"]
        probabilities = {key: float(value) for key, value in answer["probabilities"].items()}
        allowed = set(ACTIONS[language])
        if action not in allowed or set(probabilities) != allowed:
            raise ValueError("Invalid action or probability labels")
        if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
                or abs(sum(probabilities.values()) - 1) > .001):
            raise ValueError("Invalid probability vector")
        if output.get("warnings") or any(a.get("truncated") for a in output["answers"].values()):
            raise ValueError("Model reported a warning or truncated input")
        tokens = int(output.get("usage", {}).get("input_tokens", 0))
        if tokens < 0:
            raise ValueError("Invalid input token count")
        confidence = answer.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid model confidence")
        return dict(action=action, probabilities=probabilities,
                    confidence=confidence, input_tokens=tokens)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Malformed model answer") from exc


class PublicBackend:
    def __init__(self, spec, device="auto", threads=4):
        import torch

        torch.set_num_threads(threads)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.spec = spec
        if spec.backend == "quyet":
            import quyet

            self.model = quyet.load(spec.source, device=self.device, revision=spec.revision)
            package_version = quyet.__version__
        elif spec.backend == "laya":
            try:
                import laya
            except ImportError as exc:
                raise RuntimeError("The optional comparison needs the installed laya SDK") from exc
            path = Path(spec.source).expanduser()
            if not path.is_dir():
                from huggingface_hub import snapshot_download

                # Laya's load() does not accept revision. Resolve a pinned public
                # snapshot ourselves, then pass that directory to its public API.
                prefix = f"{spec.subfolder}/" if spec.subfolder else ""
                path = Path(snapshot_download(spec.source, revision=spec.revision,
                    allow_patterns=[prefix + pattern for pattern in
                                    ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")]))
            self.model = laya.load(str(path), subfolder=spec.subfolder, device=self.device, fast=False)
            package_version = laya.__version__
        else:
            raise ValueError("Unknown backend")
        self.meta = dict(spec.portable_metadata(), device=self.device, threads=threads,
                         package_version=package_version, torch_version=str(torch.__version__),
                         python=platform.python_version(), platform=platform.system(),
                         architecture=platform.machine(),
                         hardware=torch.cuda.get_device_name(0) if self.device == "cuda" else
                         (platform.processor() or platform.machine()),
                         runtime="native PyTorch; no compilation or optional fast path")

    def predict(self, state, questions):
        if self.device == "cuda":
            import torch
            torch.cuda.synchronize()
        output = self.model.predict(state, questions, strict=True) if self.spec.backend == "quyet" else \
            self.model.predict(state, questions)
        if self.device == "cuda":
            torch.cuda.synchronize()
        if self.spec.backend == "laya":
            # The native SDK can clip inputs without returning a truncation flag.
            max_len = getattr(self.model, "cfg", {}).get("max_len")
            if max_len and output.get("usage", {}).get("input_tokens", 0) >= max_len:
                raise ValueError("Laya input reached its context limit")
        return output

    def close(self):
        # Each race owns a disposable process, which releases CPU/GPU allocations.
        self.model = None
