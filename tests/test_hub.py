import json

import pytest

from quyet import hub


def test_read_config_rejects_non_quyet_dirs(tmp_path):
    with pytest.raises(ValueError, match="quyet_config.json is missing"):
        hub.read_config(tmp_path)
    (tmp_path / "quyet_config.json").write_text(json.dumps({"quyet_format": 2, "kind": "encoder"}))
    with pytest.raises(ValueError, match="upgrade quyet"):
        hub.read_config(tmp_path)
    (tmp_path / "quyet_config.json").write_text(json.dumps({"quyet_format": 1, "kind": "vision"}))
    with pytest.raises(ValueError, match="kind"):
        hub.read_config(tmp_path)


def test_resolve_local_dir(tmp_path):
    assert hub.resolve(str(tmp_path)) == tmp_path
