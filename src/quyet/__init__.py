"""Quyet decision models. quyet.load(name_or_path) -> model with .predict(state, questions)."""
from .errors import QuestionError
from .hub import load

__version__ = "1.0.2"
__all__ = ["QuestionError", "load", "__version__"]
