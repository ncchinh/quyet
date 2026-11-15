"""Small Gemma text student with PLE and cross-layer KV sharing; scratch only."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_student(root, small):
    import torch
    from transformers import Gemma4TextConfig, Gemma4TextModel
    from safetensors.torch import save_file
    from quyet.student.model import Student
    root = Path(root);root.mkdir(parents=True, exist_ok=True)
    folder = root/'model';folder.mkdir()
    cfg = json.loads((small/'quyet_config.json').read_text())
    for name in ('tokenizer.json','tokenizer_config.json'):
        shutil.copyfile(small/name, folder/name)
    text = Gemma4TextConfig(vocab_size=262144, vocab_size_per_layer_input=262144,
        hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, global_head_dim=16, attention_k_eq_v=False,
        layer_types=['sliding_attention','full_attention']*3, sliding_window=16,
        hidden_size_per_layer_input=4, num_kv_shared_layers=2, max_position_embeddings=32768,
        use_cache=False, use_bidirectional_attention='vision')
    text._attn_implementation='sdpa'
    torch.manual_seed(5272)
    model = Student(Gemma4TextModel(text)).to(dtype=torch.bfloat16)
    state = model.state_dict();keys = list(state)
    cfg['weights'] = {}
    for i, selected in enumerate((keys[:len(keys)//2], keys[len(keys)//2:])):
        name=f'weights-{i:03d}.safetensors'
        save_file({k:state[k].contiguous() for k in selected}, str(folder/name))
        cfg['weights'][name]={'size':(folder/name).stat().st_size,'sha256':sha(folder/name)}
    (folder/'config.json').write_text(json.dumps({'text_config':text.to_dict()})+'\n')
    cfg['files']={name:sha(folder/name) for name in cfg['files']}
    cfg['name']='Synthetic-Student-Test'
    (folder/'quyet_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    reference=root/'reference';reference.mkdir()
    for name in cfg['weights']:(reference/name).hardlink_to(folder/name)
    (reference/'MANIFEST.json').write_text(json.dumps({'format':'q12-student-text-weights-1','dtype':'bfloat16',
        'files':{k:v['sha256'] for k,v in cfg['weights'].items()}})+'\n')
    requests = [{'state':{'message':'Please close my card, I lost it yesterday.'}, 'questions':{
        'intent':{'type':'choice','instructions':'What does the customer want?',
                  'criteria':{'cancel':'close the card','limit':'change the limit','other':None}},
        'urgent':{'type':'noul','instructions':'The request is urgent.'},
        'mood':{'type':'score','instructions':'How upset is the customer?','criteria':['calm','annoyed','angry']}}},
        {'state':{'message':'Please close my card at the café.'},'questions':{
        'intent':{'type':'choice','instructions':'Choose an action.','criteria':['Close the card','Wait']}}}]
    (root/'requests.json').write_text(json.dumps(requests,ensure_ascii=False,indent=2)+'\n')
    return folder,requests
