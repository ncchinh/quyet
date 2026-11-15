"""F32 CPU correctness at sliding boundaries, with mixed readouts and LoRA."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from quyet.adapter import prefix_cache as cache_module
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration
from quyet.adapter.model import DecisionModel
from quyet.adapter.runtime import forward
from quyet.adapter.prefix_cache import PrefixCache, cached_records, prefix_cache_bytes, shared_prefix_length, VERIFIED_TRANSFORMERS_VERSION


@pytest.fixture(scope='module')
def verified_cache():
    if cache_module.transformers.__version__ != VERIFIED_TRANSFORMERS_VERSION:
        pytest.skip(f'Prefix-cache numerical tests require transformers=={VERIFIED_TRANSFORMERS_VERSION}; found {cache_module.transformers.__version__}')


def test_unverified_version_rejected(monkeypatch):
    monkeypatch.setattr(cache_module.transformers, '__version__', 'unverified')
    with pytest.raises(ValueError) as error:
        PrefixCache(None, [3])
    assert str(error.value) == f'Prefix caching requires transformers=={VERIFIED_TRANSFORMERS_VERSION}; found unverified'


@pytest.fixture(scope='module', params=[False, True])
def tiny_model(request):
    torch.manual_seed(419);torch.set_num_threads(4)
    cfg=Gemma4TextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=6,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,global_head_dim=16,
        num_global_key_value_heads=1,attention_k_eq_v=True,
        layer_types=['sliding_attention','full_attention']*3,sliding_window=16,
        num_kv_shared_layers=2,hidden_size_per_layer_input=0,max_position_embeddings=1024,
        use_bidirectional_attention='vision',use_cache=False)
    cfg._attn_implementation='sdpa'
    base=Gemma4ForConditionalGeneration(Gemma4Config(text_config=cfg,vision_config=None,audio_config=None))
    if request.param:
        pytest.importorskip('peft')
        from peft import LoraConfig,get_peft_model
        base=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],
                                          task_type='CAUSAL_LM',init_lora_weights=False))
    model=DecisionModel(base).eval().requires_grad_(False)
    with torch.no_grad():model.option_head.weight.normal_(std=.1)
    return model


def requests(length,count):
    prefix=[3+i%20 for i in range(length)]
    rr=[]
    for i in range(count):
        n=9+3*i;kind=('choice','noul','score')[i%3];k=3 if kind!='noul' else 2
        rr.append({'question_id':str(i),'input_ids':prefix+[40+i]+[61]*(n-1),
            'decision':{'decision_type':kind,'readout':'head' if kind=='choice' else 'letter',
                'positions':list(range(length+n-k,length+n)) if kind=='choice' else [length+n-1],
                'symbol_ids':[] if kind=='choice' else list(range(90,90+k)),
                'option_ids':['true','false'] if kind=='noul' else list(map(str,range(k)))}})
    return rr


@pytest.mark.parametrize('length',[8,16,41])
@pytest.mark.parametrize('count',[2,3,4,5,6,7,8,9])
def test_batched_suffix_matches(verified_cache,tiny_model,length,count):
    rr=requests(length,count);model=tiny_model
    assert shared_prefix_length(rr,1)==length
    expected=forward(model,rr)
    cfg=model.prefix_cache_decoder().config
    size=prefix_cache_bytes(cfg,length,4)
    for budget in (8*1024**3,4*size):  # at most two suffixes in each forced batch
        actual,c=cached_records(model,[rr],1,max_cache_bytes=budget,max_batch_rows=9)
        for q,z in zip(rr,expected):
            got=actual[0,q['question_id']]
            np.testing.assert_allclose(got,z,atol=1e-5,rtol=0)
            assert np.argmax(got)==np.argmax(z)
        batches=1 if budget>4*size else (count+1)//2
        assert c['forward_passes']==1+batches and c['suffix_batches']==batches
        assert c['tokens_skipped']==(count-1)*length
        assert c['tokens_processed']==sum(len(q['input_ids']) for q in rr)-c['tokens_skipped']
        assert c['peak_prefix_copy_bytes']<=budget


@pytest.mark.parametrize('length',[8,16,41])
def test_immutable_order_and_budget(verified_cache,tiny_model,length):
    rr=requests(length,9);model=tiny_model
    prefix=PrefixCache(model.prefix_cache_decoder(),rr[0]['input_ids'][:length])
    saved=deepcopy(prefix._cache)
    actual_bytes=sum(t.numel()*t.element_size() for l in saved.layers for t in (l.keys,l.values))
    assert actual_bytes==prefix_cache_bytes(model.prefix_cache_decoder().config,length,4)
    a=prefix.suffix_hidden_batch(rr);b=prefix.suffix_hidden_batch(list(reversed(rr)))
    for x,y in zip(a,reversed(b)):torch.testing.assert_close(x,y,atol=1e-5,rtol=0)
    for old,new in zip(saved.layers,prefix._cache.layers):
        assert old.get_seq_length()==new.get_seq_length()
        assert torch.equal(old.keys,new.keys) and torch.equal(old.values,new.values)
    value,c=cached_records(model,[rr],1,max_cache_bytes=2*actual_bytes-1)
    assert not value and c['records_over_cache_budget']==1 and c['forward_passes']==0
    _,c=cached_records(model,[rr],1,max_batch_rows=3,max_batch_tokens=1)
    assert c['suffix_batches']==9


def test_invalid_budget(tiny_model):
    with pytest.raises(ValueError,match='positive integers'):
        cached_records(tiny_model,[requests(8,2)],1,max_cache_bytes=0)
