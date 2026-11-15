import json

import pytest

from quyet import hub


def test_read_config_rejects_non_quyet_dirs(tmp_path):
    with pytest.raises(ValueError, match="quyet_config.json is missing"):
        hub.read_config(tmp_path)
    (tmp_path / "quyet_config.json").write_text(json.dumps({"quyet_format": 3, "kind": "encoder"}))
    with pytest.raises(ValueError, match="upgrade quyet"):
        hub.read_config(tmp_path)
    (tmp_path / "quyet_config.json").write_text(json.dumps({"quyet_format": 1, "kind": "vision"}))
    with pytest.raises(ValueError, match="kind"):
        hub.read_config(tmp_path)


def test_resolve_local_dir(tmp_path):
    assert hub.resolve(str(tmp_path)) == tmp_path

"""Offline repository-id loads using the real Hugging Face cache resolver."""
import gc
import hashlib
import os
import importlib.util
import json
from pathlib import Path
import socket

import pytest
import torch
import quyet

REVISION = 'a' * 40


def support(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def snapshot(cache, repo, source, documents=True):
    root = cache/('models--'+repo.replace('/', '--'))
    folder = root/'snapshots'/REVISION
    (root/'blobs').mkdir(parents=True);(root/'refs').mkdir()
    folder.mkdir(parents=True)
    (root/'refs/main').write_text(REVISION)
    for path in sorted(source.rglob('*')):
        if not path.is_file():continue
        blob = root/'blobs'/hashlib.sha256(str(path.relative_to(source)).encode()).hexdigest()
        blob.symlink_to(path.resolve())
        target = folder/path.relative_to(source);target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(blob, target.parent))
    if documents:
        for name in ('README.md', 'LICENSE', 'NOTICE', '.gitattributes'):
            if not (folder/name).exists():
                (folder/name).write_text('Repository metadata.\n')
        (folder/'.cache/huggingface').mkdir(parents=True)
        (folder/'.cache/huggingface/download.metadata').write_text('local download metadata')
    return folder


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from huggingface_hub import constants
    folder = tmp_path/'hub'
    monkeypatch.setattr(constants, 'HF_HUB_CACHE', str(folder))
    monkeypatch.setattr(constants, 'HF_HUB_OFFLINE', True)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    def forbidden(*args, **kwargs):pytest.fail('Network access attempted')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    return folder


@pytest.fixture(scope='module')
def fixtures(tmp_path_factory, model_root):
    large = model_root/'Quyet-1.2-Large';small = model_root/'Quyet-1.2-Small'
    if not (large/'quyet_config.json').is_file() or not (small/'quyet_config.json').is_file():
        pytest.skip('Release tokenizers and bindings are not available')
    root = tmp_path_factory.mktemp('hub-fixtures')
    base, adapter, requests = support('support').build_synthetic(root/'large', large)
    cfg = json.loads((adapter/'quyet_config.json').read_text())
    cfg['base'].update(model_id='test/base', revision=REVISION)
    (adapter/'quyet_config.json').write_text(json.dumps(cfg))
    student, _ = support('student_support').build_student(root/'small', small)
    legacy = [support('support').build_legacy_llm(root, base, large, version) for version in (1,2)]
    return base, adapter, student, legacy, requests


def test_student_repository_metadata(cache, fixtures, tmp_path):
    # Regular files isolate the published-repository metadata defect from symlinks.
    import shutil
    from quyet.student.validation import verify_package
    folder = tmp_path/'regular';shutil.copytree(fixtures[2], folder)
    for name in ('README.md', 'LICENSE', 'NOTICE', '.gitattributes'):(folder/name).write_text('metadata')
    (folder/'.cache').mkdir();(folder/'.cache/metadata').write_text('download')
    assert verify_package(folder, json.loads((folder/'quyet_config.json').read_text()))['weight_sizes_verified']==2


@pytest.mark.parametrize('documents', [False, True])
def test_student_snapshot(cache, fixtures, documents):
    snapshot(cache, 'test/student', fixtures[2], documents=documents)
    model = quyet.load('test/student', device='cpu', dtype=torch.float32)
    assert model.predict(**fixtures[4][0])['answers']


@pytest.mark.parametrize('local_base', [False, True])
@pytest.mark.parametrize('revision', [None, REVISION])
def test_adapter_snapshot(cache, fixtures, local_base, revision):
    base = snapshot(cache, 'test/base', fixtures[0])
    snapshot(cache, 'test/adapter', fixtures[1])
    model = quyet.load('test/adapter', revision=revision, device='cpu', dtype=torch.float32,
                       **({'base':base} if local_base else {}))
    assert model.predict_many(fixtures[4])[0]['answers']


def test_cached_base_without_explicit_path(cache, fixtures):
    snapshot(cache, 'test/base', fixtures[0])
    model = quyet.load(fixtures[1], device='cpu', dtype=torch.float32)
    assert model.predict(**fixtures[4][0])['answers']


@pytest.mark.parametrize('version', [1,2])
def test_legacy_llm_snapshot(cache, fixtures, version):
    snapshot(cache, 'test/legacy', fixtures[3][version-1])
    model = quyet.load('test/legacy', device='cpu', dtype=torch.float32)
    assert model.predict(**fixtures[4][0])['answers']


@pytest.mark.parametrize('kind', ['Tiny', 'Small', 'Large'])
def test_real_release_snapshot(cache, model_root, kind):
    source = model_root/('Quyet-1.2-'+kind)
    if not (source/'quyet_config.json').is_file():pytest.skip('Real release folder is absent')
    folder = snapshot(cache, 'test/release', source)
    if kind=='Large':
        from quyet.hub import resolve, read_config
        cfg=read_config(folder)
        if cfg['kind']=='merged':
            from quyet.merged.validation import verify_package
        else:
            from quyet.adapter.validation import verify_package
        assert resolve('test/release')==folder
        assert verify_package(folder, cfg)['alphabet_symbols_verified']==256
    else:
        model = quyet.load('test/release', device='cpu', **({'dtype':torch.float32} if kind=='Small' else {}))
        result = model.predict('The door is open.', {'q':{'type':'choice', 'instructions':'Choose.', 'criteria':['Open','Closed']}})
        assert result['answers']['q']['type']=='choice'
        del model;gc.collect()


@pytest.mark.parametrize('kind', ['adapter','student'])
@pytest.mark.parametrize('case', ['missing','tampered','parent'])
def test_snapshot_rejects_bad_listed_file(cache, fixtures, kind, case):
    source = fixtures[1 if kind=='adapter' else 2]
    folder = snapshot(cache, 'test/bad', source)
    name = 'tokenizer_config.json'
    if case=='parent':
        cfg=json.loads((folder/'quyet_config.json').read_text())
        cfg['files']['../outside.json']=cfg['files'].pop(name)
        (folder/'quyet_config.json').unlink();(folder/'quyet_config.json').write_text(json.dumps(cfg))
    else:
        (folder/name).unlink()
        if case=='tampered':(folder/name).write_text('{}')
    with pytest.raises(ValueError):quyet.load('test/bad', device='cpu', **({'base':fixtures[0]} if kind=='adapter' else {}))


@pytest.mark.parametrize('suffix', ['.safetensors','.bin','.pt','.pth','.ckpt','.gguf','.onnx'])
def test_unexpected_nested_weight(cache, fixtures, suffix):
    folder = snapshot(cache, 'test/student', fixtures[2])
    extra=folder/'.cache/nested';extra.mkdir(parents=True)
    (extra/('extra'+suffix)).write_bytes(b'weight')
    with pytest.raises(ValueError, match='unexpected=.*extra'):
        quyet.load('test/student', device='cpu')


@pytest.mark.parametrize('name', ['../a','nested/../../a','/a',r'nested\a','C:/a'])
def test_unsafe_configured_name(tmp_path, name):
    from quyet.adapter.validation import local_file
    with pytest.raises(ValueError, match='Invalid package file path'):local_file(tmp_path, name)


@pytest.mark.parametrize('case', ['missing','size'])
@pytest.mark.parametrize('local_base', [False, True])
def test_snapshot_base_shard_checks(cache, fixtures, case, local_base):
    base = snapshot(cache, 'test/base', fixtures[0])
    snapshot(cache, 'test/adapter', fixtures[1])
    cfg = json.loads((fixtures[1]/'quyet_config.json').read_text())
    shard = base/next(iter(cfg['base']['shards']))
    shard.unlink()
    if case=='size':shard.write_bytes(b'truncated')
    with pytest.raises(ValueError, match='missing base shard|Base shard size mismatch'):
        quyet.load('test/adapter', device='cpu', **({'base':base} if local_base else {}))
