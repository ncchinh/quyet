import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
import quyet
from quyet.adapter.runtime import forward
from quyet.student.validation import verify_package, verify_weights


@pytest.fixture(scope='module')
def student(tmp_path_factory, model_root):
    small=model_root/'Quyet-1.2-Small'
    if not (small/'quyet_config.json').is_file():pytest.skip('Small package not available')
    spec=importlib.util.spec_from_file_location('student_support',Path(__file__).with_name('student_support.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    folder,requests=mod.build_student(tmp_path_factory.mktemp('student'),small)
    return quyet.load(folder,device='cpu'),folder,requests


def test_public_and_offline(student,monkeypatch):
    model,folder,requests=student
    def forbidden(*a,**kw):pytest.fail('Network resolution attempted')
    monkeypatch.setattr('huggingface_hub.snapshot_download',forbidden)
    monkeypatch.setattr('huggingface_hub.hf_hub_download',forbidden)
    loaded=quyet.load(folder,device='cpu')
    responses=loaded.predict_many(requests);raw=loaded.raw_probabilities(requests)
    assert responses==model.predict_many(requests)
    assert set(responses[0])=={'model','answers','warnings','usage'}
    assert set(responses[0]['answers'])=={'intent','urgent','mood'}
    assert responses[0]['warnings']==[] and all('truncated' not in a for a in responses[0]['answers'].values())
    for row in raw:
        for values in row.values():assert sum(values)==pytest.approx(1)
    assert loaded.verification['tensors_consumed']==len(loaded.model.state_dict())


@pytest.mark.parametrize('kind,k,accept',[('choice',2,True),('choice',10,True),('choice',11,True),('choice',60,True),('choice',256,True),('choice',257,False),('score',10,True),('score',11,False)])
def test_limits(student,kind,k,accept):
    model,_,_=student
    question={'type':kind,'instructions':'Choose.','criteria':[str(i) for i in range(k)]}
    if accept:
        result=model.predict('Short state.',{'q':question})
        assert len(result['answers']['q']['probabilities'])==k
    else:
        with pytest.raises(quyet.QuestionError,match='supports at most'):model.predict('s',{'q':question})


@pytest.mark.parametrize('strict',[False,True])
def test_no_truncation(student,strict):
    model,_,requests=student
    with patch.object(model.renderer.head,'window',16):
        with pytest.raises(quyet.QuestionError,match='does not truncate'):model.predict(**requests[0],strict=strict)


@pytest.mark.parametrize('option,value,message',[
    ('prefix_cache_max_bytes',0,'positive integer'),('prefix_cache_max_bytes',True,'positive integer'),
    ('base','unused','self-contained'),('device_map','auto','one device'),('dtype','float16','dtype must')])
def test_rejected_runtime_options(student,option,value,message):
    _,folder,_=student
    with pytest.raises(ValueError,match=message):quyet.load(folder,device='cpu',**{option:value})


@pytest.mark.parametrize('case',['metadata','size','hash','alphabet','file_set'])
def test_integrity(student,case,tmp_path):
    model,folder,_=student;cfg=copy.deepcopy(model.config)
    if case=='metadata':
        cfg['files']['config.json']='0'*64
        with pytest.raises(ValueError,match='metadata hash mismatch'):verify_package(folder,cfg)
    elif case=='size':
        next(iter(cfg['weights'].values()))['size']+=1
        with pytest.raises(ValueError,match='weight size mismatch'):verify_package(folder,cfg)
    elif case=='hash':
        next(iter(cfg['weights'].values()))['sha256']='0'*64
        with pytest.raises(ValueError,match='weight SHA256 mismatch'):verify_weights(folder,cfg)
    elif case=='alphabet':
        cfg['renderer']['reference_symbol_ids'][-1]=123
        with pytest.raises(ValueError,match='alphabet binding'):verify_package(folder,cfg)
    else:
        with pytest.raises(ValueError,match='file set differs'):verify_package(tmp_path,cfg)


def test_full_hashes_and_dependency(student,monkeypatch):
    model,folder,_=student
    from quyet.student import verify_weights as public_verify
    assert public_verify(folder)['weight_hashes_verified']==2
    monkeypatch.setitem(sys.modules,'accelerate',None)
    with pytest.raises(ImportError,match=r'quyet\[student\]'):quyet.load(folder,device='cpu')


def test_header_validation(student,tmp_path):
    from safetensors.torch import load_file,save_file
    import shutil
    _,folder,_=student
    dest=tmp_path/'broken';shutil.copytree(folder,dest)
    cfg=json.loads((dest/'quyet_config.json').read_text());name=next(iter(cfg['weights']))
    state=load_file(str(dest/name));state.pop(next(iter(state)));save_file(state,str(dest/name))
    cfg['weights'][name]['size']=(dest/name).stat().st_size
    (dest/'quyet_config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='Missing student tensors'):quyet.load(dest,device='cpu')


def test_renderer_and_batching(student):
    model,_,requests=student
    q={'type':'choice','instructions':'Choose.','criteria':{'a':None,'b':False}}
    a=model.renderer.request_public('s',q,'q')
    b=model.renderer.request_public('s',{**q,'criteria':['a','b']},'q')
    assert a==b and a['decision']['symbol_ids']==[]
    rendered=[model.renderer.request_public(r['state'],q,k) for r in requests for k,q in r['questions'].items()]
    batch=forward(model.model,rendered);single=[forward(model.model,[r])[0] for r in rendered]
    assert max(np.max(abs(np.array(a)-b)) for a,b in zip(batch,single))<1e-5
    assert all(np.argmax(a)==np.argmax(b) for a,b in zip(batch,single))


@pytest.mark.parametrize('cache',[False,True])
def test_cli(student,tmp_path,capsys,cache):
    from quyet.cli import main
    _,folder,requests=student
    path=tmp_path/'requests.jsonl';path.write_text(json.dumps(requests[0])+'\n')
    assert main(['predict',str(folder),'--device','cpu','--input',str(path)]+(['--prefix-cache'] if cache else []))==0
    assert json.loads(capsys.readouterr().out)['model']=='Synthetic-Student-Test'


@pytest.mark.parametrize('question',[
    {'type':'choice','instructions':'','criteria':['a','b']},
    {'type':'choice','instructions':'Choose.','criteria':['a','a']},
    {'type':'score','instructions':'Rate.','criteria':['one']},
    {'type':'noul','instructions':'True?','criteria':[]},
    {'type':'unknown','instructions':'Choose.'}, None])
def test_question_validation(student,question):
    model,_,_=student
    with pytest.raises(quyet.QuestionError):model.predict('s',{'q':question})


def test_qualified_context_boundary(student):
    model,_,_=student
    q={'type':'choice','instructions':'Choose.','criteria':['a','b']}
    n=23400
    with patch.object(model.renderer.head,'window',32768):
        initial=model.renderer.request_public('x '*n,q,'q')
        n+=23434-len(initial['input_ids'])
    exact=model.renderer.request_public('x '*n,q,'q')
    assert len(exact['input_ids'])==23434
    assert model.renderer.request_public('x '*(n+1),q,'q')['unsupported']=='over_window'


def test_public_prefix_cache(student):
    model,path,requests=student
    requests=copy.deepcopy(requests)
    requests[0]['state']={'message':'Please close my card. '*30}
    import numpy as np
    expected=model.raw_probabilities(requests)
    cached=quyet.load(path,device='cpu',prefix_cache=True)
    actual=cached.raw_probabilities(requests)
    for a,b in zip(actual,expected):
        for key in a:
            np.testing.assert_allclose(a[key],b[key],atol=1e-5,rtol=0)
            assert np.argmax(a[key])==np.argmax(b[key])
    assert cached.cache_counters['records_using_cache']==1
    assert cached.predict_many(requests)==model.predict_many(requests)
    fallback=quyet.load(path,device='cpu',prefix_cache=True,prefix_cache_max_bytes=1)
    assert fallback.raw_probabilities(requests)==expected
    assert fallback.cache_counters['records_over_cache_budget']==1


def test_student_cache_version_guard(student,monkeypatch):
    import transformers
    monkeypatch.setattr(transformers,'__version__','0.0.0')
    with pytest.raises(ValueError,match='requires transformers==5.17.0'):
        quyet.load(student[1],device='cpu',prefix_cache=True)
