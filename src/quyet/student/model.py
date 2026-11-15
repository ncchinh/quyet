"""Text-only Student readouts and causal right padding from the evaluated path."""
import torch
from torch import nn
from torch.nn import functional as F
from ..adapter.validation import check


class Student(nn.Module):
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.option_head = nn.Linear(text.config.hidden_size, 1)

    def forward_batch(self, requests, *, compute_loss=False):
        check(not compute_loss, 'This runtime is inference-only')
        check(bool(requests), 'Empty microbatch')
        device = self.text.embed_tokens.weight.device
        width = max(len(r.input_ids) for r in requests)
        ids = torch.tensor([r.input_ids + [0] * (width - len(r.input_ids)) for r in requests], device=device)
        hidden = self.text(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           position_ids=torch.arange(width, device=device)[None]).last_hidden_state
        results = []
        for i, request in enumerate(requests):
            logits = []
            for d in request.decisions:
                if d['readout'] == 'head':
                    h = hidden[i, d['positions']].float().to(self.option_head.weight.device)
                    z = F.linear(h, self.option_head.weight.float(), self.option_head.bias.float()).squeeze(-1)
                else:
                    projection = self.text.embed_tokens.weight
                    symbols = torch.tensor(d['symbol_ids'], device=projection.device)
                    z = F.linear(hidden[i, d['positions'][0]].to(projection.device), projection[symbols]).float()
                check(len(z) == len(d['option_ids']), 'Option/readout width differs')
                logits.append(z.clone())
            results.append((hidden.new_empty((0,), dtype=torch.float32), logits))
        return results

    def prefix_cache_decoder(self):
        return self.text

    def prefix_cache_readout(self, hidden, decision):
        if decision['readout'] == 'head':
            h = hidden[decision['positions']].float().to(self.option_head.weight.device)
            z = F.linear(h, self.option_head.weight.float(), self.option_head.bias.float()).squeeze(-1)
        else:
            projection = self.text.embed_tokens.weight
            symbols = torch.tensor(decision['symbol_ids'], device=projection.device)
            z = F.linear(hidden[decision['positions'][0]].to(projection.device), projection[symbols]).float()
        check(len(z) == len(decision['option_ids']), 'Option/readout width differs')
        return z.clone()
