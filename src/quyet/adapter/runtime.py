"""Format-2 local adapter/head plus a pinned Gemma base, using evaluated readouts."""
import gc
import math
from pathlib import Path

from ..base import QuyetModel, Result, check_state
from ..calibration import temperature, uncalibrated_warnings
from ..errors import QuestionError
from ..questions import parse_questions, option_limits
from .render import Renderer
from .request import Request
from .validation import check, verify_package, verify_base


def batches(flat, budget, batch_size):
    current=[]; width=0
    for item in sorted(flat,key=lambda x:len(x[2]['input_ids'])):
        n=len(item[2]['input_ids'])
        if current and (max(n,width)*(len(current)+1)>budget or len(current)>=batch_size):
            yield current;current=[];width=0
        current.append(item);width=max(n,width)
    if current:yield current


def probabilities(logits,t):
    check(math.isfinite(t) and t>0,'Invalid temperature')
    check(bool(logits) and all(math.isfinite(z) for z in logits),'Invalid logits')
    z=[v/t for v in logits];top=max(z)
    values=[math.exp(v-top) for v in z];total=sum(values)
    return [v/total for v in values]


def forward(model,requests):
    import torch
    try:
        with torch.inference_mode():
            result=model.forward_batch([Request(r['input_ids'],[r['decision']]) for r in requests],compute_loss=False)
            return [x[1][0].float().cpu().tolist() for x in result]
    except torch.cuda.OutOfMemoryError:
        if len(requests)==1:raise
        gc.collect();torch.cuda.empty_cache();middle=len(requests)//2
        return forward(model,requests[:middle])+forward(model,requests[middle:])


class AdapterModel(QuyetModel):
    kind='adapter'

    def __init__(self,path,config,device,dtype=None,device_map=None,base=None,prefix_cache=False,prefix_cache_max_bytes=8 * 1024 ** 3):
        import torch
        from transformers import AutoModelForCausalLM
        from huggingface_hub.constants import HF_HUB_OFFLINE
        from .model import DecisionModel
        self.path,self.config=Path(path),config
        self.verification=verify_package(self.path,config)
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError('The adapter model requires peft; install "quyet[adapter]"') from e
        self.name=config['name'];self.max_options=option_limits(config)
        self.renderer=Renderer(self.path,config)
        self.temps=dict(config.get('temperatures') or {})
        self.max_batch_tokens=config['batching']['max_tokens'];self.max_batch_rows=config['batching']['max_rows']
        self.prefix_cache=prefix_cache
        check(type(prefix_cache_max_bytes) is int and prefix_cache_max_bytes>0, 'prefix_cache_max_bytes must be a positive integer')
        self.prefix_cache_max_bytes=prefix_cache_max_bytes
        if prefix_cache:
            import transformers
            from .prefix_cache import VERIFIED_TRANSFORMERS_VERSION
            check(transformers.__version__==VERIFIED_TRANSFORMERS_VERSION,
                  f'Prefix caching requires transformers=={VERIFIED_TRANSFORMERS_VERSION}; found {transformers.__version__}')
        load_options=dict(dtype=dtype or torch.bfloat16,attn_implementation=config['attn_implementation'],use_safetensors=True)
        load_options['device_map']=device_map if device_map is not None else {'':str(device)}
        if base is None:
            # Resolve and verify the pinned metadata through transformers before loading shards.
            from transformers.utils.hub import cached_file
            from .validation import sha
            for name,expected in config['base']['files'].items():
                try:
                    found=cached_file(config['base']['model_id'],name,revision=config['base']['revision'],local_files_only=HF_HUB_OFFLINE)
                except OSError as e:
                    if HF_HUB_OFFLINE:
                        raise ValueError('Pinned Large base weights are not bundled. Offline mode requires base=<local pinned weights directory> or the pinned base in the Hub cache; no download attempted') from e
                    raise
                check(sha(found)==expected,'Base metadata hash mismatch: '+name)
            if HF_HUB_OFFLINE:
                verify_base(Path(found).parent,config)
            backbone=AutoModelForCausalLM.from_pretrained(config['base']['model_id'],
                        revision=config['base']['revision'],local_files_only=HF_HUB_OFFLINE,**load_options)
        else:
            base=verify_base(base,config)
            backbone=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,**load_options)
        backbone.config.use_cache=False
        backbone=PeftModel.from_pretrained(backbone,str(self.path/'adapter'),is_trainable=False,local_files_only=True)
        self.model=DecisionModel(backbone)
        from safetensors.torch import load_file
        self.model.option_head.load_state_dict(load_file(str(self.path/config['option_head']['file'])),strict=True)
        self.model.option_head.to(backbone.get_output_embeddings().weight.device)
        self.model.eval().requires_grad_(False)

    def _run(self,requests,strict):
        prepared=[];flat=[]
        for i,r in enumerate(requests):
            check_state(r['state'])
            decisions=parse_questions(r['questions'],self.max_options)
            rendered={}
            questions={str(k):q for k,q in r['questions'].items()}
            for key,d in decisions.items():
                # Parse the original public criteria, then use the evaluated converter,
                # which deliberately supplies fixed true/false texts for noul.
                q=questions[key]
                if d.type=='choice' and isinstance(q['criteria'],list):
                    q={**q,'criteria':dict(zip(d.labels,d.labels))}
                item=self.renderer.request_public(r['state'],q,key)
                if 'unsupported' in item:
                    reason=item['unsupported']
                    if reason=='over_window':
                        raise QuestionError(f"question {key!r}: prompt exceeds {self.config['limits']['max_input_tokens']} complete tokens; this model does not truncate")
                    raise QuestionError(f'question {key!r}: unsupported input ({reason})')
                check(item['decision']['option_ids']==list(d.labels),'Rendered option order differs from public labels')
                rendered[key]=item;flat.append((i,key,item))
            prepared.append((decisions,rendered))
        values={}
        if self.prefix_cache:
            from .prefix_cache import cached_records
            values,self.cache_counters=cached_records(self.model,[list(r.values()) for _,r in prepared],
                max_cache_bytes=self.prefix_cache_max_bytes,max_batch_tokens=self.max_batch_tokens,max_batch_rows=self.max_batch_rows)
        for batch in batches(flat,self.max_batch_tokens,self.max_batch_rows):
            pending=[(i,k,q) for i,k,q in batch if (i,k) not in values]
            if pending:
                z=forward(self.model,[q for _,_,q in pending])
                check(len(z)==len(pending),'Forward row count differs')
                values.update({(i,k):v for (i,k,_),v in zip(pending,z)})
        results=[]
        for i,(decisions,rendered) in enumerate(prepared):
            probs={key:probabilities(values[i,key],temperature(self.temps,d.type,len(d.labels))) for key,d in decisions.items()}
            results.append(Result(decisions,probs,{k:False for k in decisions},
                           sum(len(q['input_ids']) for q in rendered.values()),uncalibrated_warnings(decisions,self.temps)))
        return results
