"""Small temporary Gemma/LoRA fixture; never uses real base weights."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_synthetic(root,large_folder):
    import torch
    from transformers import Gemma4TextConfig,Gemma4Config,Gemma4ForConditionalGeneration
    from tokenizers import Tokenizer
    from peft import LoraConfig,get_peft_model
    from safetensors.torch import save_file
    torch.manual_seed(2319);torch.set_num_threads(4)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    base=root/'base';model=root/'model'
    base.mkdir();model.mkdir()
    cfg=json.loads((Path(large_folder)/'quyet_config.json').read_text())
    for name in ('tokenizer.json','tokenizer_config.json'):
        shutil.copyfile(Path(large_folder)/name,model/name)
    vocab=Tokenizer.from_file(str(model/'tokenizer.json')).get_vocab_size(with_added_tokens=True)
    text=Gemma4TextConfig(vocab_size=vocab,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,global_head_dim=16,
            num_global_key_value_heads=1,attention_k_eq_v=True,
            layer_types=['sliding_attention','full_attention']*2,sliding_window=16,
            hidden_size_per_layer_input=0,num_kv_shared_layers=0,max_position_embeddings=32768,
            use_cache=False,use_bidirectional_attention='vision')
    text._attn_implementation='sdpa'
    backbone=Gemma4ForConditionalGeneration(Gemma4Config(text_config=text,vision_config=None,audio_config=None))
    backbone.save_pretrained(base,max_shard_size='10MB',safe_serialization=True)
    assert (base/'model.safetensors.index.json').is_file()
    backbone=get_peft_model(backbone,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
                         target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',init_lora_weights=False))
    backbone.save_pretrained(model/'adapter',safe_serialization=True)
    head={'weight':torch.randn(1,32)*.1,'bias':torch.randn(1)*.1}
    save_file(head,str(model/'option_head.safetensors'))
    cfg['option_head']['in_features']=32
    cfg['base']['files']={n:sha(base/n) for n in ('config.json','model.safetensors.index.json')}
    cfg['base']['shards']={p.name:{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(base.glob('*.safetensors'))}
    cfg.pop('weights',None);cfg.pop('calibration',None)
    cfg['kind']='adapter';cfg['adapter']={'config':'adapter/adapter_config.json','weights':'adapter/adapter_model.safetensors'}
    cfg['files']={n:sha(model/n) for n in ('adapter/adapter_config.json','adapter/adapter_model.safetensors','option_head.safetensors','tokenizer.json','tokenizer_config.json')}
    cfg['name']='Synthetic-Adapter-Test'
    (model/'quyet_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    requests=[{'state':'The gate is closed. '+'A shared state with several details. '*12,'questions':{
       'choice':{'type':'choice','instructions':'Choose an action.','criteria':{'open':'Open the gate','wait':'Wait','leave':'Leave'}},
       'noul':{'type':'noul','instructions':'The gate is open.','criteria':{'true':'ignored true','false':'ignored false'}},
       'score':{'type':'score','instructions':'Rate the urgency.','criteria':['Low','Medium','High']}}},
       {'state':{'place':'The café is closed.'},'questions':{
       'choice':{'type':'choice','instructions':'Choose an action.','criteria':{'a':'Open','b':'Wait'}}}}]
    (root/'requests.json').write_text(json.dumps(requests,ensure_ascii=False,indent=2)+'\n')
    return base,model,requests


def unit_requests():
    return [{'state':state,'questions':{
        'choice':{'type':'choice','instructions':'Choose.','criteria':{'a':None,'b':{'place':'Café closed'}}},
        'noul':{'type':'noul','instructions':'The gate is open.','criteria':{'false':'Closed','true':'Open'}},
        'score':{'type':'score','instructions':'Rate.','criteria':['Low','Medium','High']}}}
        for state in ('The gate is closed.',{'place':'The café is closed.'},['Earlier','Later'])]


def llm_identity():
    """Exercise both frozen prompt versions, batching and warnings with a deterministic fake LM."""
    import torch
    from types import SimpleNamespace
    from quyet.llm.runtime import LLMModel
    class Tokenizer:
        def __call__(self,text,**kw):return {'input_ids':[ord(c) for c in text]}
        def decode(self,ids):return ''.join(chr(i) for i in ids)
        def apply_chat_template(self,msgs,**kw):return json.dumps(msgs,ensure_ascii=False)+' assistant:'
    class Model:
        def __call__(self,input_ids,attention_mask,**kw):
            s=(input_ids*attention_mask).sum(1).float()
            return SimpleNamespace(logits=torch.sin(s[:,None,None]+torch.arange(16)[None,None,:]))
    out=[]
    requests=unit_requests()+[{'state':['HEAD','x'*2000,'TAIL'],'questions':unit_requests()[0]['questions']}]
    for version in (1,2):
        model=LLMModel.__new__(LLMModel);model.name='legacy-llm';model.tok=Tokenizer();model.model=Model()
        model.device='cpu';model.letter_ids=list(range(10));model.temps={'choice':1.2,'noul':.9,'score':1.1}
        model.prompt_version=version;model.max_state_tokens=80;model.max_prompt_tokens=2000;model.min_state_tokens=10
        model.max_batch_tokens=8000;model.pad_id=0;model.max_options={'choice':10,'score':10,'noul':2}
        responses=model.predict_many(requests);raw=model.raw_probabilities(requests)
        try:model.predict(**requests[-1],strict=True)
        except Exception as e:error=type(e).__name__+': '+str(e)
        out.append({'version':version,'responses':responses,'raw':raw,'strict_error':error})
    return out


def build_legacy_llm(root,base,large_folder,version):
    folder=Path(root)/('legacy-v'+str(version));folder.mkdir(exist_ok=True)
    for path in Path(base).iterdir():
        if path.is_file():shutil.copyfile(path,folder/path.name)
    for name in ('tokenizer.json','tokenizer_config.json'):
        shutil.copyfile(Path(large_folder)/name,folder/name)
    tokenizer_config=json.loads((folder/'tokenizer_config.json').read_text())
    tokenizer_config['chat_template']="{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:"
    (folder/'tokenizer_config.json').write_text(json.dumps(tokenizer_config)+'\n')
    cfg={'quyet_format':1,'kind':'llm','name':'Legacy-Synthetic-LLM','prompt_version':version,
         'temperatures':{'choice':1.2,'noul':.9,'score':1.1},
         'limits':{'max_state_tokens':80,'max_prompt_tokens':2000,'min_state_tokens':10}}
    (folder/'quyet_config.json').write_text(json.dumps(cfg)+'\n')
    return folder
