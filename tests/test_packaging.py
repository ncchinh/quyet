import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# internal paths must never ship (split so this file does not contain them); .superpowers files are kept out by the allowlist
LEAK_MARKERS = ("/ho" + "me/", "/ro" + "ot/", "/tm" + "p/", "/work" + "space/")
SDIST_ALLOWED = ("src/quyet/", "examples/gem_run/", "tests/", "README.md", "LICENSE", "NOTICE", "pyproject.toml", "PKG-INFO", ".gitignore")
WHEEL_ALLOWED = ("quyet/", "quyet-1.0.0.dist-info/")


def _check(files, allowed):
    stray = [n for n in files if not n.startswith(allowed)]
    leaks = [(n, m) for n, data in files.items() for m in LEAK_MARKERS if m.encode() in data]
    assert not stray, f"unexpected files in the distribution: {stray[:10]}"
    assert not leaks, f"internal markers in the distribution: {leaks[:10]}"


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build the distributions")
def test_built_distributions_ship_only_package_files(tmp_path):
    subprocess.run(["uv", "build", "--out-dir", str(tmp_path)], cwd=ROOT, check=True, capture_output=True)
    with tarfile.open(next(tmp_path.glob("quyet-*.tar.gz"))) as t:
        files = {m.name.split("/", 1)[1]: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
        _check(files, SDIST_ALLOWED)
        for name in ("replay.py", "run.py", "engine.py", "backends.py", "README.md",
                     "web/template.html", "web/assets/world.png", "recordings/sample.json.gz"):
            assert "examples/gem_run/" + name in files
        assert not any(n.endswith((".mp4", ".webm", ".pyc")) for n in files)
    with zipfile.ZipFile(next(tmp_path.glob("quyet-*.whl"))) as z:
        _check({n: z.read(n) for n in z.namelist()}, WHEEL_ALLOWED)


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build the distributions")
def test_notice_and_metadata_ship(tmp_path):
    subprocess.run(["uv", "build", "--out-dir", str(tmp_path)], cwd=ROOT, check=True, capture_output=True)
    with tarfile.open(next(tmp_path.glob("quyet-*.tar.gz"))) as t:
        assert any(m.name.endswith("/NOTICE") for m in t.getmembers())
    with zipfile.ZipFile(next(tmp_path.glob("quyet-*.whl"))) as z:
        names = z.namelist()
        assert any(n.endswith(".dist-info/licenses/NOTICE") for n in names)
        meta = z.read(next(n for n in names if n.endswith("METADATA"))).decode()
    assert "Project-URL: Source, https://github.com/ncchinh/quyet" in meta and "Author: Chinh Nguyen" in meta


def test_readme_has_no_internal_names():
    bad = [line for line in (ROOT / "README.md").read_text().splitlines()
           if "la" + "ya" in line.lower() and "Convai" not in line]
    assert not bad, bad


def test_sources_have_no_internal_project_name():
    bad = [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py") if "bao" + "cong" in p.read_text().lower()]
    bad += [n for n in ("README.md", "NOTICE") if "bao" + "cong" in (ROOT / n).read_text().lower()]
    assert not bad, bad
