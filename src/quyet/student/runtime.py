"""Self-contained student weights; reuse the format-2 public inference interface."""
import json
from pathlib import Path
from ..adapter.runtime import AdapterModel
from .render import Renderer
from ..adapter.validation import check
from ..questions import option_limits
from .validation import verify_package


class StudentModel(AdapterModel):
    kind = 'student'

    def __init__(self, path, config, device, dtype=None, device_map=None, base=None,
                 prefix_cache=False, prefix_cache_max_bytes=8 * 1024 ** 3):
        import torch
        from transformers import Gemma4TextConfig, Gemma4TextModel
        from safetensors import safe_open
        from safetensors.torch import load_file
        from .model import Student
        check(base is None, 'Student weights are self-contained; base must be omitted')
        check(type(prefix_cache_max_bytes) is int and prefix_cache_max_bytes > 0,
              'prefix_cache_max_bytes must be a positive integer')
        if prefix_cache:
            import transformers
            from ..adapter.prefix_cache import VERIFIED_TRANSFORMERS_VERSION
            check(transformers.__version__ == VERIFIED_TRANSFORMERS_VERSION,
                  f'Prefix caching requires transformers=={VERIFIED_TRANSFORMERS_VERSION}; found {transformers.__version__}')
        check(device_map is None, 'Student currently supports one device; use device= instead of device_map')
        self.path, self.config = Path(path), config
        self.verification = verify_package(self.path, config)
        try:
            from accelerate import init_empty_weights
        except ImportError as e:
            raise ImportError('The student model requires accelerate; install "quyet[student]"') from e
        self.name = config['name'];self.max_options = option_limits(config)
        self.renderer = Renderer(self.path, config)
        self.temps = dict(config['temperatures'])
        self.max_batch_tokens = config['batching']['max_tokens'];self.max_batch_rows = config['batching']['max_rows']
        self.prefix_cache = prefix_cache
        self.prefix_cache_max_bytes = prefix_cache_max_bytes
        cfg = Gemma4TextConfig(**json.loads((self.path/'config.json').read_text())['text_config'])
        cfg._attn_implementation = config['attn_implementation'];cfg.use_cache = False
        check(getattr(cfg, 'use_bidirectional_attention', None) != 'all', 'Student requires causal text attention')
        with init_empty_weights():
            self.model = Student(Gemma4TextModel(cfg))
        expected = self.model.state_dict();seen = set()
        # Validate every header before allocating weights: assign=True alone does
        # not detect omissions in a split checkpoint.
        for name in config['weights']:
            with safe_open(str(self.path/name), framework='pt', device='cpu') as f:
                for key in f.keys():
                    check(key in expected and key not in seen, 'Unexpected or duplicate student tensor: ' + key)
                    part = f.get_slice(key)
                    check(part.get_shape() == list(expected[key].shape), 'Student tensor shape mismatch: ' + key)
                    check(part.get_dtype() == 'BF16', 'Student tensor dtype mismatch: ' + key)
                    seen.add(key)
        check(seen == set(expected), 'Missing student tensors: ' + str(sorted(set(expected)-seen)))
        del expected
        target = dtype or (getattr(torch, config['dtype']) if torch.device(device).type == 'cuda' else torch.float32)
        if isinstance(target, str):target = getattr(torch, target, None)
        check(target in (torch.float32, torch.bfloat16), 'Student dtype must be float32 or bfloat16')
        for name in config['weights']:
            state = load_file(str(self.path/name), device='cpu')
            state = {k:v.to(device=device, dtype=target) for k,v in state.items()}
            self.model.load_state_dict(state, strict=False, assign=True)
            del state
        self.model.to(device=device, dtype=target).eval().requires_grad_(False)
        check(not any(p.is_meta for p in self.model.parameters()), 'Student has unmaterialized parameters')
        self.verification['tensors_consumed'] = len(seen)
