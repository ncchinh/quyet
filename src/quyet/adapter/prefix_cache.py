"""Record-local, inference-only exact prefix reuse for Gemma 4 text requests.

One prefill, then bounded right-padded batches of decision suffixes. No option branching,
cross-record cache, or crop/rollback. Tested against transformers 5.17.0.
"""
from copy import deepcopy

import torch
import transformers
from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

from .validation import check

VERIFIED_TRANSFORMERS_VERSION = '5.17.0'
DEFAULT_CACHE_BYTES = 8 * 1024 ** 3


def shared_prefix_length(requests, min_prefix_tokens=32):
    """Bound the exact common prefix by the earliest readout (exclusive)."""
    check(type(min_prefix_tokens) is int and min_prefix_tokens > 0,
          'min_prefix_tokens must be a positive integer')
    if len(requests) < 2:
        return 0
    limit = min(min(q['decision']['positions']) for q in requests)
    first = requests[0]['input_ids']
    length = 0
    for tokens in zip(*(q['input_ids'] for q in requests)):
        if length >= limit or any(t != first[length] for t in tokens):
            break
        length += 1
    return length if length >= min_prefix_tokens else 0


class PrefixCache:
    """An immutable stored prefill, bound to one decoder and exact token IDs.

    Every suffix receives a deep copy, including sliding cumulative lengths.
    The instance is discarded at the end of a record, before any model change.
    """

    @torch.inference_mode()
    def __init__(self, decoder, input_ids):
        check(transformers.__version__ == VERIFIED_TRANSFORMERS_VERSION,
              f'Prefix caching requires transformers=={VERIFIED_TRANSFORMERS_VERSION}; found {transformers.__version__}')
        check(not decoder.training, 'Prefix caching requires eval mode')
        cfg = decoder.config.get_text_config()
        check(cfg.model_type == 'gemma4_text', 'Prefix caching requires Gemma 4 text')
        check(getattr(cfg, 'is_causal', True) and cfg.use_bidirectional_attention != 'all',
              'Prefix caching requires causal text attention')
        check(cfg._attn_implementation in ('sdpa', 'eager'),
              'Prefix caching supports sdpa/eager attention only')
        check(all(p['rope_type'] in ('default', 'proportional') for p in cfg.rope_parameters.values()),
              'Prefix caching requires fixed rotary frequencies')
        check(bool(input_ids), 'Empty cached prefix')
        self.decoder = decoder
        self.input_ids = tuple(input_ids)
        self.device = next(decoder.parameters()).device
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        # Supply the configured cache BEFORE the multimodal wrapper builds masks.
        # cache_utils.py:1721-1741 selects full/sliding layers and omits KV-sharing
        # followers. Gemma4Model forwards this same object to its text model.
        cache = DynamicCache(config=cfg)
        check(all(type(layer) in (DynamicLayer, DynamicSlidingWindowLayer) for layer in cache.layers),
              'Unsupported prefix cache layer')
        output = decoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                         position_ids=torch.arange(len(input_ids), device=self.device)[None],
                         past_key_values=cache, use_cache=True)
        check(output.past_key_values is cache, 'Decoder replaced the prefix cache')
        # Sliding updates retain slices of the full prefill storage (5.17.0
        # cache_utils.py:251-263). Compact those views before storing/copying.
        for layer in cache.layers:
            layer.keys = layer.keys.clone()
            layer.values = layer.values.clone()
        self._cache = cache

    @torch.inference_mode()
    def suffix_hidden_batch(self, requests):
        """Return suffix hidden states in input order; stored prefix stays immutable."""
        length = len(self.input_ids)
        check(bool(requests), 'Empty suffix batch')
        check(not self.decoder.training, 'Prefix caching requires eval mode')
        for request in requests:
            check(tuple(request['input_ids'][:length]) == self.input_ids, 'Cached token prefix differs')
            check(length <= min(request['decision']['positions']), 'Cached prefix includes a readout')
            check(length < len(request['input_ids']), 'Empty decision suffix')
        sizes = [len(q['input_ids']) - length for q in requests]
        width = max(sizes)
        pad = self.decoder.config.get_text_config().pad_token_id or 0
        ids = torch.tensor([q['input_ids'][length:] + [pad] * (width - n)
                            for q, n in zip(requests, sizes)], dtype=torch.long, device=self.device)
        # 5.17.0 cache_utils.py:1633-1636 dispatches to each layer;
        # :192-196 repeats K/V on dim 0, preserving sliding cumulative_length.
        # Deep-copy first: neither expansion nor suffix updates touch the prefill.
        cache = deepcopy(self._cache)
        if len(requests) > 1:
            cache.batch_repeat_interleave(len(requests))
        output = self.decoder(
            input_ids=ids,
            attention_mask=torch.ones((len(requests), length + width), dtype=torch.long, device=self.device),
            position_ids=torch.arange(length, length + width, device=self.device)[None],
            past_key_values=cache, use_cache=True)
        # All-valid right padding is strictly after every real token of its row.
        return [output.last_hidden_state[i, :n] for i, n in enumerate(sizes)]

    def suffix_hidden(self, request):
        """Internal singleton reference, also used by the layer diagnostic."""
        return self.suffix_hidden_batch([request])[0]


def prefix_cache_bytes(config, length, element_size):
    """Retained single-row K/V bytes, excluding KV-sharing follower layers.

    Sliding caches retain min(prefix, window-1); full caches retain the prefix.
    This is a prefix-copy budget, not a bound on activations or suffix K/V.
    """
    cfg = config.get_text_config()
    count = cfg.num_hidden_layers - getattr(cfg, 'num_kv_shared_layers', 0)
    total = 0
    for kind, layer in zip(cfg.layer_types[:count], cfg.per_layer_config[:count]):
        tokens = min(length, cfg.sliding_window - 1) if kind == 'sliding_attention' else length
        total += 2 * layer.num_key_value_heads * layer.head_dim * tokens * element_size
    return total


def suffix_batches(requests, length, prefix_bytes, *, max_cache_bytes=DEFAULT_CACHE_BYTES,
                   max_batch_tokens=8192, max_batch_rows=8):
    """Plan stable length-sorted batches, including stored/copy/expansion overlap.

    Return no batches when even one copied prefix cannot fit: the caller uses
    normal uncached batching for that record. Oversize singleton suffixes follow
    the ordinary runtime's token-budget convention.
    """
    for value in (max_cache_bytes, max_batch_tokens, max_batch_rows):
        check(type(value) is int and value > 0, 'Cache and batch limits must be positive integers')
    if 2 * prefix_bytes > max_cache_bytes:
        return
    capacity = max(1, max_cache_bytes // max(1, prefix_bytes) - 2)
    capacity = min(capacity, max_batch_rows)
    batch = []
    for q in sorted(requests, key=lambda q: len(q['input_ids'])):
        width = len(q['input_ids']) - length
        if batch and (len(batch) >= capacity or width * (len(batch) + 1) > max_batch_tokens):
            yield batch
            batch = []
        batch.append(q)
    if batch:
        yield batch


@torch.inference_mode()
def cached_records(model, grouped_requests, min_prefix_tokens=32, *,
                   max_cache_bytes=DEFAULT_CACHE_BYTES, max_batch_tokens=8192, max_batch_rows=8):
    """Return cached logits and counters; ineligible records remain for fallback."""
    check(type(min_prefix_tokens) is int and min_prefix_tokens > 0,
          'min_prefix_tokens must be a positive integer')
    for value in (max_cache_bytes, max_batch_tokens, max_batch_rows):
        check(type(value) is int and value > 0, 'Cache and batch limits must be positive integers')
    values = {}
    counters = dict(requests=sum(map(len, grouped_requests)), records_using_cache=0,
                    prefix_tokens_computed_once=0, tokens_skipped=0, forward_passes=0,
                    suffix_batches=0, tokens_processed=0, padded_tokens_processed=0,
                    records_over_cache_budget=0, peak_prefix_copy_bytes=0)
    for index, requests in enumerate(grouped_requests):
        length = shared_prefix_length(requests, min_prefix_tokens)
        if not length:
            continue
        check(not model.training, 'Prefix caching requires eval mode')
        decoder = model.prefix_cache_decoder()
        size = prefix_cache_bytes(decoder.config, length, next(decoder.parameters()).element_size())
        planned = list(suffix_batches(requests, length, size, max_cache_bytes=max_cache_bytes,
                                      max_batch_tokens=max_batch_tokens, max_batch_rows=max_batch_rows))
        if not planned:
            counters['records_over_cache_budget'] += 1
            continue
        prefix = PrefixCache(decoder, requests[0]['input_ids'][:length])
        for batch in planned:
            hidden = prefix.suffix_hidden_batch(batch)
            for q, h in zip(batch, hidden):
                decision = {**q['decision'], 'positions': [p - length for p in q['decision']['positions']]}
                z = model.prefix_cache_readout(h, decision)
                values[index, q['question_id']] = z.float().cpu().tolist()
            counters['padded_tokens_processed'] += len(batch) * max(len(q['input_ids']) - length for q in batch)
            copies = 2 if len(batch) == 1 else len(batch) + 2
            counters['peak_prefix_copy_bytes'] = max(counters['peak_prefix_copy_bytes'], copies * size)
            del hidden, h, z
        del prefix
        counters['records_using_cache'] += 1
        counters['prefix_tokens_computed_once'] += length
        counters['tokens_skipped'] += (len(requests) - 1) * length
        counters['forward_passes'] += 1 + len(planned)
        counters['suffix_batches'] += len(planned)
        counters['tokens_processed'] += length + sum(len(q['input_ids']) - length for q in requests)
        counters['padded_tokens_processed'] += length
    return values, counters
