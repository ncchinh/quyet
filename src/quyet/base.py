"""Shared model interface: predict / predict_many / raw_probabilities over Jev-format requests."""
from __future__ import annotations

from dataclasses import dataclass, field

from .answers import build_response
from .errors import QuestionError


@dataclass
class Result:
    decisions: dict          # qid -> Decision
    probs: dict              # qid -> list[float] aligned with Decision.labels
    truncated: dict          # qid -> bool
    input_tokens: int
    warnings: list = field(default_factory=list)


def check_state(state):
    if not isinstance(state, (str, dict, list)):
        raise QuestionError(f"state must be a string, object or list, got {type(state).__name__}")


def truncation_warning(state, total, kept):
    return {"code": "state_truncated", "state_tokens": int(total), "kept_tokens": int(kept),
            "kept": "tail" if isinstance(state, list) else "head"}


class QuyetModel:
    name = "quyet"
    kind = ""

    def _run(self, requests, strict):
        raise NotImplementedError

    def predict(self, state, questions, *, strict=False):
        """One request -> {"model", "answers", "usage", "warnings"}. strict=True raises instead of truncating."""
        return self.predict_many([{"state": state, "questions": questions}], strict=strict)[0]

    def predict_many(self, requests, *, strict=False):
        """Several requests batched together; any invalid request raises QuestionError for the whole call."""
        return [build_response(self.name, r.decisions, r.probs, r.truncated, r.input_tokens, r.warnings)
                for r in self._run(list(requests), strict)]

    def raw_probabilities(self, requests, *, strict=False):
        """Unrounded probabilities per question id, aligned with the answer labels."""
        return [r.probs for r in self._run(list(requests), strict)]
