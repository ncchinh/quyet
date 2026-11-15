"""Load merged weights locally; all inference is inherited from the adapter runtime."""
from pathlib import Path
from ..adapter.runtime import AdapterModel
from ..adapter.render import Renderer
from ..adapter.validation import check
from ..questions import option_limits
from .validation import verify_package


class MergedModel(AdapterModel):
    kind = 'merged'

    def __init__(self,path,config,device,dtype=None,device_map=None,base=None,prefix_cache=False,prefix_cache_max_bytes=8 * 1024 ** 3):
        import torch
        from transformers import AutoModelForCausalLM
        from ..adapter.model import DecisionModel
        check(base is None, 'Merged weights are self-contained; base must be omitted')
        self.path,self.config=Path(path),config
        self.verification=verify_package(self.path,config)
        self.name=config['name'];self.max_options=option_limits(config)
        self.renderer=Renderer(self.path,config)
        self.temps=dict(config.get('temperatures') or {})
        self.max_batch_tokens=config['batching']['max_tokens'];self.max_batch_rows=config['batching']['max_rows']
        self.prefix_cache=prefix_cache
        check(type(prefix_cache_max_bytes) is int and prefix_cache_max_bytes>0, 'prefix_cache_max_bytes must be a positive integer')
        self.prefix_cache_max_bytes=prefix_cache_max_bytes
        if prefix_cache:
            import transformers
            from ..adapter.prefix_cache import VERIFIED_TRANSFORMERS_VERSION
            check(transformers.__version__==VERIFIED_TRANSFORMERS_VERSION,
                  f'Prefix caching requires transformers=={VERIFIED_TRANSFORMERS_VERSION}; found {transformers.__version__}')
        load_options=dict(dtype=dtype or torch.bfloat16,attn_implementation=config['attn_implementation'],use_safetensors=True)
        load_options['device_map']=device_map if device_map is not None else {'':str(device)}
        backbone=AutoModelForCausalLM.from_pretrained(str(self.path),local_files_only=True,**load_options)
        backbone.config.use_cache=False
        self.model=DecisionModel(backbone)
        from safetensors.torch import load_file
        self.model.option_head.load_state_dict(load_file(str(self.path/config['option_head']['file'])),strict=True)
        self.model.option_head.to(backbone.get_output_embeddings().weight.device)
        self.model.eval().requires_grad_(False)

