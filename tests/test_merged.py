"""Merged loading changes; public inference is the existing adapter implementation."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch
import quyet
from quyet.adapter.runtime import AdapterModel, forward
from quyet.adapter.validation import sha
from quyet.merged.runtime import MergedModel
from quyet.merged.validation import verify_package, verify_weights


def helper(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def merge_fixture(root, adapter, base):
    """FP32 PEFT arithmetic is the synthetic reference, not the release merge algorithm."""
    model=quyet.load(adapter,base=base,device='cpu',dtype=torch.float32)
    folder=Path(root);folder.mkdir(parents=True)
    backbone=model.model.backbone.merge_and_unload(safe_merge=True)
    backbone.save_pretrained(folder,max_shard_size='10MB',safe_serialization=True)
    for name in ('tokenizer.json','tokenizer_config.json','option_head.safetensors'):
        shutil.copyfile(Path(adapter)/name,folder/name)
    cfg=copy.deepcopy(model.config);cfg['kind']='merged';cfg.pop('adapter')
    cfg['base']['download_required']=False
    cfg['weights']={p.name:{'size':p.stat().st_size,'sha256':sha(p)} for p in folder.glob('model*.safetensors')}
    cfg['files']={n:sha(folder/n) for n in ('config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','option_head.safetensors')}
    cfg['calibration']={'fitted_on':'base_plus_adapter','merged_gpu_acceptance_recorded':False}
    (folder/'quyet_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    return folder


@pytest.fixture(scope='module')
def merged(tmp_path_factory,model_root):
    root=tmp_path_factory.mktemp('merged')
    base,adapter,requests=helper('support').build_synthetic(root/'fixture',model_root/'Quyet-1.2-Large')
    folder=merge_fixture(root/'merged',adapter,base)
    return base,adapter,folder,requests


def test_shared_inference_methods():
    for name in ('_run','predict','predict_many','raw_probabilities'):
        assert getattr(MergedModel,name) is getattr(AdapterModel,name)


@pytest.mark.parametrize('cached',[False,True])
def test_public_merged_adapter_parity(merged,cached):
    base,adapter,folder,requests=merged
    original=quyet.load(adapter,base=base,device='cpu',dtype=torch.float32,prefix_cache=cached)
    model=quyet.load(folder,device='cpu',dtype=torch.float32,prefix_cache=cached)
    expected=original.predict_many(requests);actual=model.predict_many(requests)
    assert actual==expected
    a=original.raw_probabilities(requests);b=model.raw_probabilities(requests)
    assert max(abs(np.asarray(a[i][k])-b[i][k]).max() for i in range(len(a)) for k in a[i])<1e-5
    rr=[model.renderer.request_public(r['state'],q,k) for r in requests for k,q in r['questions'].items()]
    x=forward(original.model,rr);y=forward(model.model,rr)
    delta=max(float(np.max(abs(np.asarray(u)-v))) for u,v in zip(x,y))
    assert delta<1e-5 and all(np.argmax(u)==np.argmax(v) for u,v in zip(x,y))
    print('MERGED FP32 cached',cached,'max_abs_logit_difference',delta)


def test_merged_cache_parity_and_budget(merged):
    _,_,folder,requests=merged
    model=quyet.load(folder,device='cpu',dtype=torch.float32)
    expected=model.predict_many(requests)
    model.prefix_cache=True
    assert model.predict_many(requests)==expected
    assert model.cache_counters['records_using_cache']==1
    model.prefix_cache_max_bytes=1
    assert model.predict_many(requests)==expected
    assert model.cache_counters['records_using_cache']==0


def test_no_peft_or_download(merged,monkeypatch):
    import socket
    monkeypatch.setitem(sys.modules,'peft',None)
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k:pytest.fail('Network attempted'))
    model=quyet.load(merged[2],device='cpu',dtype=torch.float32,device_map={'':'cpu'})
    assert model.predict_many(merged[3])[0]['answers']


def test_merged_hub_snapshot(merged,tmp_path,monkeypatch):
    from huggingface_hub import constants
    import socket
    folder=helper('test_hub').snapshot(tmp_path/'hub','test/merged',merged[2])
    monkeypatch.setattr(constants,'HF_HUB_CACHE',str(tmp_path/'hub'))
    monkeypatch.setattr(constants,'HF_HUB_OFFLINE',True)
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k:pytest.fail('Network attempted'))
    model=quyet.load('test/merged',revision='a'*40,device='cpu',dtype=torch.float32,prefix_cache=True)
    assert model.path==folder and model.predict_many(merged[3])[0]['answers']


@pytest.mark.parametrize('case',['missing','metadata','size','hash','extra','parent','inventory','adapter_config'])
def test_integrity_failures(merged,tmp_path,case):
    folder=tmp_path/'model';shutil.copytree(merged[2],folder)
    cfg=json.loads((folder/'quyet_config.json').read_text());name=next(iter(cfg['weights']))
    if case=='missing':(folder/name).unlink()
    elif case=='metadata':(folder/'config.json').write_text('{}')
    elif case=='size':cfg['weights'][name]['size']+=1
    elif case=='hash':cfg['weights'][name]['sha256']='0'*64
    elif case=='adapter_config':(folder/'adapter_config.json').write_text('{}')
    elif case=='extra':(folder/'extra.pth').write_bytes(b'bad')
    elif case=='parent':cfg['weights']['../'+name]=cfg['weights'].pop(name)
    elif case=='inventory':
        index=json.loads((folder/'model.safetensors.index.json').read_text());index['weight_map']['missing.tensor']=name
        (folder/'model.safetensors.index.json').write_text(json.dumps(index));cfg['files']['model.safetensors.index.json']=sha(folder/'model.safetensors.index.json')
    with pytest.raises(ValueError):verify_weights(folder,cfg) if case=='hash' else verify_package(folder,cfg)


def test_full_hashes_and_base_refused(merged):
    folder=merged[2];cfg=json.loads((folder/'quyet_config.json').read_text())
    assert verify_weights(folder,cfg)['weight_hashes_verified']==len(cfg['weights'])
    with pytest.raises(ValueError,match='self-contained; base must be omitted'):
        quyet.load(folder,base=merged[0],device='cpu')


def test_merged_limits_and_cli(merged,tmp_path,capsys):
    import quyet.cli
    folder=merged[2];model=quyet.load(folder,device='cpu',dtype=torch.float32)
    for kind,k in [('choice',257),('score',11)]:
        with pytest.raises(quyet.QuestionError):model.predict('state',{'q':{'type':kind,'instructions':'Choose.','criteria':list(range(k))}})
    model.renderer.head.window=5
    with pytest.raises(quyet.QuestionError,match='does not truncate'):model.predict_many(merged[3])
    input_file=tmp_path/'input.jsonl';input_file.write_text(json.dumps(merged[3][0])+'\n')
    assert quyet.cli.main(['predict',str(folder),'--device','cpu','--prefix-cache','--input',str(input_file)])==0
    assert json.loads(capsys.readouterr().out)['answers']
