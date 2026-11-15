import json
from pathlib import Path
import pytest

from quyet import QuestionError
from quyet.questions import option_limits,parse_question,parse_questions
from quyet.calibration import uncalibrated_warnings


def choice(k):
    return {'type':'choice','instructions':'Choose.','criteria':list(range(k))}


@pytest.mark.parametrize('k',[2,10,11,60,255,256])
def test_declared_choice_limits(k):
    limits=option_limits({'limits':{'max_options':{'choice':256,'score':10,'noul':2}}})
    d=parse_question(choice(k),limits)
    assert d.labels==tuple(str(i) for i in range(k))
    assert d.texts==(None,)*k
    if k<=10:assert d==parse_question(choice(k))
    else:
        with pytest.raises(QuestionError,match='choice supports at most 10 options'):
            parse_question(choice(k))


@pytest.mark.parametrize('q,message',[
    (choice(257),'choice supports at most 256 options, got 257'),
    ({'type':'score','instructions':'Rate.','criteria':['x']*11},'score supports at most 10 options, got 11'),
    ({**choice(11),'criteria':[1,'1',*range(2,11)]},'choice labels must be unique'),
    ({**choice(11),'instructions':''},'instructions must be a non-empty string'),
    ([], 'a question must be an object with type, instructions and criteria'),
    (choice(1),'choice needs at least 2 options'),
])
def test_error_class_and_frozen_style(q,message):
    with pytest.raises(QuestionError) as err:
        parse_questions({'qid':q},{'choice':256,'score':10,'noul':2})
    assert str(err.value)=="question 'qid': "+message


@pytest.mark.parametrize('limits',[{'choice':257},{'choice':True},{'score':11},{'noul':3},{'unknown':2}])
def test_invalid_config_limits(limits):
    with pytest.raises(ValueError):option_limits({'limits':{'max_options':limits}})


def test_model_limits_and_warning_fallback():
    assert option_limits({})=={'choice':10,'score':10,'noul':2}
    assert option_limits({'limits':{'max_options':{'choice':2}}})['choice']==2
    with pytest.raises(ValueError):option_limits({'limits':{'max_options':{'choice':11}}},choice_ceiling=10)
    decisions=parse_questions({'small':choice(10),'wide':choice(11)},{'choice':256,'score':10,'noul':2})
    assert uncalibrated_warnings(decisions,{})==[
        {'code':'uncalibrated_option_count','question':'wide','options':11,'temperature':1.0}]
    assert uncalibrated_warnings(decisions,{'choice':1.0})==[]
    assert uncalibrated_warnings(decisions,{'choice:11+':2.0})==[]


def test_descriptions_and_noul_identity():
    criteria={i:v for i,v in enumerate([None,'',False,0,{},[],{'place':'Zürich'},['a'],1,2,3])}
    d=parse_question({**choice(11),'criteria':criteria},{'choice':256,'score':10,'noul':2})
    assert d.texts[:6]==(None,)*6 and d.texts[6:] == ('{"place": "Zürich"}','["a"]','1','2','3')
    assert parse_question({'type':'noul','instructions':'True?','criteria':{'extra':'ignored'}}).labels==('true','false')


@pytest.fixture(scope='module')
def tiny(model_root):
    import quyet,torch
    torch.set_num_threads(4)
    path=model_root/'Quyet-1.2-Tiny'
    if not (path/'model.safetensors').is_file():pytest.skip('Tiny weights are not available')
    return quyet.load(path,device='cpu')


def test_real_tiny_identity(tiny,legacy_reference):
    unit=legacy_reference['unit']
    assert tiny.predict_many(unit['requests'])==unit['responses']
    assert tiny.raw_probabilities(unit['requests'])==unit['raw']
    assert tiny.max_batch_tokens==32768 and tiny.max_batch_rows==256


def test_real_tiny_256_and_no_warning_below_11(tiny):
    a=tiny.predict('The gate is closed.',{'q':choice(256)})
    assert len(a['answers']['q']['probabilities'])==256
    assert a['warnings']==[{'code':'uncalibrated_option_count','question':'q','options':256,'temperature':1.0}]
    assert tiny.predict('The gate is closed.',{'q':choice(10)})['warnings']==[]
    for q in (choice(257),{'type':'score','instructions':'Rate.','criteria':['x']*11}):
        with pytest.raises(QuestionError):tiny.predict('s',{'q':q})


def test_tiny_legacy_config_and_strict(tiny,monkeypatch):
    import numpy as np
    from quyet.encoder import inputs as I
    captured=[]
    def logits(items):
        captured.extend(items)
        return [np.zeros(len(it['markers'])) for it in items]
    monkeypatch.setattr(tiny,'_logits',logits)
    for state in ('HEAD '+' gate'*8300+' TAIL',['HEAD',' gate'*8300,'TAIL']):
        answer=tiny.predict(state,{'q':choice(2)})
        assert answer['answers']['q']['truncated']
        item=captured[-1];ids=tiny.tokens.encode(I.serialize_state(state));kept=item['kept_state_tokens']
        assert item['ids'][item['st_start']:item['st_end']]==(ids[-kept:] if isinstance(state,list) else ids[:kept])
        with pytest.raises(QuestionError,match='window .strict'):tiny.predict(state,{'q':choice(2)},strict=True)
    monkeypatch.setattr(tiny,'max_options',option_limits({}))
    with pytest.raises(QuestionError,match='at most 10'):tiny.predict('s',{'q':choice(11)})


def test_cli_legacy_behavior(tiny,monkeypatch,capsys,tmp_path):
    import quyet.cli as cli
    src=tmp_path/'requests.jsonl'
    src.write_text('\n'+json.dumps({'state':'s','questions':{'q':choice(2)}})+'\n{broken\n'+json.dumps({'state':'s','questions':{'q':choice(257)}})+'\n')
    calls=[]
    def load(*args,**kwargs):calls.append((args,kwargs));return tiny
    monkeypatch.setattr(cli,'load',load)
    assert cli.main(['predict','local-model','--input',str(src),'--device','cpu'])==1
    result=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(result)==3 and 'answers' in result[0]
    assert result[1]['error'].startswith('JSONDecodeError:')
    assert result[2]['error']=="QuestionError: question 'q': choice supports at most 256 options, got 257"
    assert calls==[(('local-model',),{'device':'cpu'})]


def test_version_and_cli_help(capsys):
    import quyet
    from quyet.cli import main
    assert quyet.__version__=='1.2.0'
    with pytest.raises(SystemExit) as exc:main(['--help'])
    assert exc.value.code==0 and 'quyet' in capsys.readouterr().out


def test_legacy_llm_identity(legacy_reference,monkeypatch):
    import importlib.util
    helper=Path(__file__).with_name('support.py')
    if not helper.is_file():pytest.skip('Source-tree compatibility helper is not available')
    spec=importlib.util.spec_from_file_location('quyet_check_data',helper)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr(mod,'unit_requests',lambda:legacy_reference['unit']['requests'])
    assert mod.llm_identity()==legacy_reference['legacy_llm']
