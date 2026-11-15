"""Quyet decision models. quyet.load(name_or_path) -> model with .predict(state, questions)."""
from .errors import QuestionError
from .hub import load

__version__ = "1.2.0"
__all__ = ["QuestionError", "load", "__version__", "verify_base_weights"]


def verify_base_weights(base, model):
    """Verify all pinned base shard hashes; model is a local model folder or config dict.

    This reads every base weight byte and can take minutes. Normal loads check
    metadata hashes and shard sizes instead.
    """
    from .hub import read_config
    from .adapter.validation import verify_base_hashes
    config=model if isinstance(model,dict) else read_config(model)
    if config.get('quyet_format')!=2 or config.get('kind')!='adapter':
        raise ValueError('Base shard verification requires a format-2 adapter model')
    return verify_base_hashes(base,config)
