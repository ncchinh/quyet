"""Self-contained merged Gemma model and optional full shard verification."""


def verify_weights(path):
    """Read every merged checkpoint byte and verify its pinned SHA256."""
    from ..hub import read_config
    from .validation import verify_weights as verify
    return verify(path, read_config(path))
