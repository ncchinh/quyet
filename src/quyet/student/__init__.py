"""Self-contained Gemma text student and optional full weight verification."""


def verify_weights(path):
    """Read every local student weight byte and verify its pinned SHA256."""
    from ..hub import read_config
    from .validation import verify_weights as verify
    return verify(path, read_config(path))
