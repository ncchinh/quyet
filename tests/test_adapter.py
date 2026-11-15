import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import quyet
from quyet.adapter.runtime import forward
from quyet.adapter.validation import verify_package


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory,model_root):
    large=model_root/'Quyet-1.2-Large'
    if not (large/'quyet_config.json').is_file():pytest.skip('Large tokenizer and binding are not available')
    path=Path(__file__).with_name('support.py')
    spec=importlib.util.spec_from_file_location('quyet_test_support',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    base,folder,requests=mod.build_synthetic(tmp_path_factory.mktemp('adapter'),large)
    model=quyet.load(folder,base=base,device='cpu',dtype=torch.float32)
    return model,base,folder,requests


def test_full_predict_and_cache(synthetic):
    model,base,folder,requests=synthetic
    assert model.verification=={'files_verified':5,'alphabet_symbols_verified':256,'head_tensors_verified':2}
    off=model.predict_many(requests);raw=model.raw_probabilities(requests)
    rendered=[[model.renderer.request_public(r['state'],q,k) for k,q in r['questions'].items()] for r in requests]
    original=[forward(model.model,rr) for rr in rendered]
    from quyet.adapter.prefix_cache import cached_records
    cached,counters=cached_records(model.model,rendered)
    maximum=0.
    for i,rr in enumerate(rendered):
        for q,z in zip(rr,original[i]):
            if (i,q['question_id']) not in cached:continue
            other=cached[i,q['question_id']]
            maximum=max(maximum,float(np.max(abs(np.array(z)-other))))
            assert np.argmax(z)==np.argmax(other)
    assert maximum<=1e-5 and counters['records_using_cache']==1
    model.prefix_cache=True
    try:
        on=model.predict_many(requests);p=model.raw_probabilities(requests)
    finally:model.prefix_cache=False
    assert off==on
    for a,b in zip(raw,p):
        for key in a:assert np.max(abs(np.array(a[key])-b[key]))<1e-5
    assert off[0]['answers']['noul']['type']=='noul'
    assert set(off[0])=={'model','answers','usage','warnings'} and off[0]['warnings']==[]
    print('SYNTHETIC PREFIX max_abs_logit_difference',maximum)


def test_hash_mismatch(synthetic):
    model,base,folder,requests=synthetic
    cfg=json.loads((folder/'quyet_config.json').read_text())
    cfg['files']['option_head.safetensors']='0'*64
    with pytest.raises(ValueError,match='Model file hash mismatch: option_head.safetensors'):
        verify_package(folder,cfg)


def test_alphabet_mismatch(synthetic):
    model,base,folder,requests=synthetic
    cfg=json.loads((folder/'quyet_config.json').read_text());cfg['renderer']['reference_symbol_ids'][-1]=123
    with pytest.raises(ValueError,match='Tokenizer alphabet binding differs'):
        verify_package(folder,cfg)


def test_missing_peft(synthetic,monkeypatch):
    model,base,folder,requests=synthetic
    monkeypatch.setitem(sys.modules,'peft',None)
    with pytest.raises(ImportError,match=r'install "quyet\[adapter\]"'):
        quyet.load(folder,base=base,device='cpu')


def test_no_offline_download(synthetic,monkeypatch,tmp_path):
    from transformers import AutoModelForCausalLM
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub.constants,'HF_HUB_CACHE',str(tmp_path/'empty-hub'))
    folder=synthetic[2]
    def forbidden(*args,**kwargs):pytest.fail('Download/base load attempted')
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',forbidden)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',forbidden)
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',forbidden)
    with pytest.raises(ValueError,match='Offline mode requires base=.*no download attempted'):
        quyet.load(folder,device='cpu')


def test_local_base_hash_mismatch(synthetic,tmp_path):
    model,base,folder,requests=synthetic
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='Base metadata hash mismatch: config.json'):
        quyet.load(folder,base=tmp_path,device='cpu')


def test_window_error_and_question_limits(synthetic,monkeypatch):
    model,base,folder,requests=synthetic
    monkeypatch.setattr(model.renderer.head,'window',5)
    with pytest.raises(quyet.QuestionError,match='complete tokens; this model does not truncate'):
        model.predict(**requests[0])
    for k,kind in ((257,'choice'),(11,'score')):
        q={'type':kind,'instructions':'Choose.','criteria':list(range(k))}
        with pytest.raises(quyet.QuestionError):model.predict('s',{'q':q})


def test_prefix_version_rejected(synthetic,monkeypatch):
    import transformers
    model,base,folder,requests=synthetic
    monkeypatch.setattr(transformers,'__version__','unverified')
    with pytest.raises(ValueError,match='Prefix caching requires transformers==5.17.0'):
        quyet.load(folder,base=base,device='cpu',prefix_cache=True)


def test_cli_new_options(synthetic,monkeypatch,tmp_path,capsys):
    import quyet.cli as cli
    model,base,folder,requests=synthetic
    src=tmp_path/'input.jsonl';src.write_text(json.dumps(requests[0])+'\n')
    calls=[]
    def load(*args,**kwargs):calls.append(kwargs);return model
    monkeypatch.setattr(cli,'load',load)
    assert cli.main(['predict','local-model','--base','base-model','--prefix-cache','--input',str(src)])==0
    assert calls==[{'device':None,'base':'base-model','prefix_cache':True}]
    assert json.loads(capsys.readouterr().out)['answers']


@pytest.mark.parametrize('version',[1,2])
def test_real_format_1_llm_identity(synthetic,model_root,legacy_llm_reference,version):
    model,base,folder,requests=synthetic
    path=Path(__file__).with_name('support.py')
    spec=importlib.util.spec_from_file_location('legacy_test_support',path)
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    legacy_folder=support.build_legacy_llm(folder.parent,base,model_root/'Quyet-1.2-Large',version)
    legacy=quyet.load(legacy_folder,device='cpu',dtype=torch.float32)
    reference=legacy_llm_reference['versions'][str(version)]
    requests=legacy_llm_reference['requests']
    assert legacy.max_batch_tokens==8000
    assert legacy.predict_many(requests)==reference['responses']
    assert legacy.raw_probabilities(requests)==reference['raw']
    with pytest.raises(quyet.QuestionError,match='at most 10'):
        legacy.predict('s',{'q':{'type':'choice','instructions':'Choose.','criteria':list(range(11))}})


@pytest.mark.parametrize('missing',[None,'',0,False,{},[]])
def test_missing_choice_descriptions_are_labels(synthetic,missing):
    model,_,_,_=synthetic
    a={'type':'choice','instructions':'Choose.','criteria':{'safe':missing,'unsafe':missing}}
    b={**a,'criteria':['safe','unsafe']}
    assert model.renderer.request_public('s',a,'q')==model.renderer.request_public('s',b,'q')


def test_compact_description_conversion(synthetic):
    model,_,_,_=synthetic
    for kind,criteria,expected in (
        ('choice',{'a':{'x':[1,True]},'b':['b',2]},['{"x":[1,true]}','["b",2]']),
        ('score',[{'x':[1,True]},['b',2]],['{"x":[1,true]}','["b",2]'])):
        q={'type':kind,'instructions':'Choose.','criteria':criteria}
        d=model.renderer.convert_question(q,'q')
        assert [o['text'] for o in d['options']]==expected


def test_full_base_hashes_and_size_errors(synthetic,tmp_path):
    import shutil
    model,base,folder,_=synthetic
    cfg=model.config
    checked=quyet.verify_base_weights(base,folder)
    assert checked['shards_verified']==len(cfg['base']['shards'])
    local=tmp_path/'base';shutil.copytree(base,local)
    name=next(iter(cfg['base']['shards']));shard=local/name
    original=shard.read_bytes()
    shard.unlink()
    from quyet.adapter.validation import verify_base
    with pytest.raises(ValueError,match='missing base shard'):
        verify_base(local,cfg)
    shard.write_bytes(original+b'x')
    with pytest.raises(ValueError,match='Base shard size mismatch'):
        verify_base(local,cfg)
    shard.write_bytes(bytes([original[0]^1])+original[1:])
    verify_base(local,cfg)  # Cheap load-time verification deliberately does not read weights.
    with pytest.raises(ValueError,match='Base shard SHA256 mismatch'):
        quyet.verify_base_weights(local,cfg)
