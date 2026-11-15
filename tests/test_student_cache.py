"""Student cache qualification: shared KV, per-layer inputs and sliding boundaries."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from transformers import Gemma4TextConfig, Gemma4TextModel
from quyet.student.model import Student
from quyet.adapter.runtime import forward
from quyet.adapter.prefix_cache import PrefixCache, cached_records, prefix_cache_bytes
from test_prefix_cache import requests, verified_cache


@pytest.fixture(scope='module', params=[6,42])
def student_cache_model(request, verified_cache):
    torch.manual_seed(5272);torch.set_num_threads(4)
    layers=request.param
    cfg=Gemma4TextConfig(vocab_size=128,vocab_size_per_layer_input=128,hidden_size=32,
        intermediate_size=64,num_hidden_layers=layers,num_attention_heads=4,num_key_value_heads=2,
        head_dim=8,global_head_dim=16,attention_k_eq_v=False,
        layer_types=['sliding_attention','full_attention']*3 if layers==6 else (['sliding_attention']*5+['full_attention'])*7,
        sliding_window=16,hidden_size_per_layer_input=8,num_kv_shared_layers=2 if layers==6 else 18,
        max_position_embeddings=2048,use_bidirectional_attention='vision',use_cache=False)
    cfg._attn_implementation='sdpa'
    return Student(Gemma4TextModel(cfg)).eval().requires_grad_(False)


@pytest.mark.parametrize('length',[15,16,17,41,513])
@pytest.mark.parametrize('count',[2,9])
def test_student_boundary_parity(student_cache_model,length,count):
    model=student_cache_model;rr=requests(length,count)
    # Include unequal suffixes longer than the window, even in a batch of two.
    for q in rr:
        q['input_ids'] += [62]*19
        q['decision']['positions']=[p+19 for p in q['decision']['positions']]
    expected=forward(model,rr);size=prefix_cache_bytes(model.text.config,length,4)
    for budget in (8*1024**3,4*size,2*size):
        actual,c=cached_records(model,[rr],1,max_cache_bytes=budget,max_batch_rows=9)
        for q,z in zip(rr,expected):
            np.testing.assert_allclose(actual[0,q['question_id']],z,atol=3e-5,rtol=0)
            assert np.argmax(actual[0,q['question_id']])==np.argmax(z)
        batches=1 if budget>4*size else (count+1)//2 if budget==4*size else count
        assert c['forward_passes']==1+batches and c['suffix_batches']==batches
        assert c['tokens_skipped']==(count-1)*length
        assert c['tokens_processed']==sum(len(q['input_ids']) for q in rr)-c['tokens_skipped']
        assert c['peak_prefix_copy_bytes']<=budget


@pytest.mark.parametrize('length',[15,16,17,41])
def test_student_immutable_budget_and_singleton(student_cache_model,length):
    model=student_cache_model;rr=requests(length,3)
    prefix=PrefixCache(model.text,rr[0]['input_ids'][:length]);saved=deepcopy(prefix._cache)
    assert len(saved.layers)==model.text.config.num_hidden_layers-model.text.config.num_kv_shared_layers
    size=sum(t.numel()*t.element_size() for layer in saved.layers for t in (layer.keys,layer.values))
    assert size==prefix_cache_bytes(model.text.config,length,4)
    a=prefix.suffix_hidden_batch(rr);b=prefix.suffix_hidden_batch(list(reversed(rr)))
    for x,y in zip(a,reversed(b)):torch.testing.assert_close(x,y,atol=3e-5,rtol=0)
    for old,new in zip(saved.layers,prefix._cache.layers):
        assert torch.equal(old.keys,new.keys) and torch.equal(old.values,new.values)
        assert old.get_seq_length()==new.get_seq_length()
    value,c=cached_records(model,[rr],1,max_cache_bytes=2*size-1)
    assert not value and c['records_over_cache_budget']==1 and c['forward_passes']==0
    value,c=cached_records(model,[rr[:1]],1)
    assert not value and c['forward_passes']==0 and c['records_using_cache']==0
    _,c=cached_records(model,[rr],1,max_batch_rows=1,max_batch_tokens=1)
    assert c['suffix_batches']==3


@pytest.mark.parametrize('length',[511,512,513])
def test_student_e4b_window_and_ple_width(student_cache_model,length):
    cfg=student_cache_model.text.config.to_dict()
    cfg.update(sliding_window=512,hidden_size_per_layer_input=256)
    cfg=Gemma4TextConfig(**cfg);cfg._attn_implementation='sdpa'
    torch.manual_seed(5272)
    model=Student(Gemma4TextModel(cfg)).eval().requires_grad_(False)
    rr=requests(length,3)
    for q in rr:
        q['input_ids'] += [62]*520
        q['decision']['positions']=[p+520 for p in q['decision']['positions']]
    expected=forward(model,rr);actual,_=cached_records(model,[rr])
    for q,z in zip(rr,expected):
        np.testing.assert_allclose(actual[0,q['question_id']],z,atol=3e-5,rtol=0)
        assert np.argmax(actual[0,q['question_id']])==np.argmax(z)
