"""Cheap local student checks, with opt-in full safetensors hashes."""
import math
from pathlib import Path
from ..adapter.validation import check, local_file, sha
from ..questions import option_limits


def verify_package(path, config):
    from tokenizers import Tokenizer
    path = Path(path)
    check(config.get('quyet_format') == 2 and config.get('kind') == 'student', 'Expected format-2 student package')
    check(isinstance(config.get('name'), str) and bool(config['name']), 'Missing model name')
    option_limits(config)
    limits = config['limits']
    check(type(limits['max_input_tokens']) is int and 0 < limits['max_input_tokens'] <= 23434
          and limits['truncate_state'] is False, 'Student inputs require a complete-token limit of at most 23434 without truncation')
    check(config['readouts'] == {'choice':'per_option_head', 'noul':'reference_alphabet', 'score':'reference_alphabet'},
          'Unsupported student readout binding')
    check(config['dtype'] == 'bfloat16' and config['attn_implementation'] == 'sdpa', 'Unsupported evaluated compute profile')
    files, weights = config['files'], config['weights']
    check(set(files) == {'config.json', 'tokenizer.json', 'tokenizer_config.json'}, 'Incomplete student metadata hashes')
    check(isinstance(weights, dict) and bool(weights), 'Missing student weight pins')
    for name, pin in weights.items():
        check('/' not in name and name.endswith('.safetensors'), 'Invalid student weight filename')
        check(type(pin['size']) is int and pin['size'] > 0, 'Invalid student weight size: ' + name)
    expected = set(files) | set(weights) | {'quyet_config.json'}
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    unexpected = {name for name in actual-expected if Path(name).suffix.lower()
                  in {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf', '.onnx'}}
    check(expected <= actual and not unexpected, 'Student model file set differs: missing=' + str(sorted(expected-actual)) + ', unexpected=' + str(sorted(unexpected)))
    for name, pin in {**{k:{'sha256':v} for k,v in files.items()}, **weights}.items():
        digest = pin['sha256']
        check(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Invalid SHA256: ' + name)
        p = local_file(path, name)
        if name in files:
            check(sha(p) == digest, 'Student metadata hash mismatch: ' + name)
        else:
            check(p.stat().st_size == pin['size'], 'Student weight size mismatch: ' + name)
    binding = config['renderer']
    check(binding['version'] == 'gemma4-w8-diet-d-v1' and binding['mode'] == 'per_option_head', 'Unknown student renderer')
    tok = Tokenizer.from_file(str(path/'tokenizer.json'))
    tok.no_truncation();tok.no_padding();tok.encode_special_tokens = False
    check(binding['delimiter_token'] == '<unused0>' and tok.token_to_id('<unused0>') == binding['delimiter_id'], 'Tokenizer delimiter binding differs')
    symbols, ids = binding['reference_symbols'], binding['reference_symbol_ids']
    check(len(symbols) == len(ids) == len(set(symbols)) == len(set(ids)) == 256, 'Missing 256-symbol alphabet binding')
    check(binding['symbols'] == list('ABCDEFGHIJ') == symbols[:10] and binding['symbol_ids'] == ids[:10], 'Alphabet/letter prefix differs')
    for symbol, token in zip(symbols, ids):
        check(type(token) is int and token >= 0 and tok.encode(symbol, add_special_tokens=False).ids == [token], 'Tokenizer alphabet binding differs: ' + symbol)
    for field in ('prefix_ids', 'suffix_ids'):
        check(isinstance(binding[field], list) and bool(binding[field]) and all(type(i) is int and i >= 0 and tok.id_to_token(i) is not None for i in binding[field]), 'Invalid chat affix: ' + field)
    check(set(config['temperatures']) == {'choice','score','noul'}, 'Missing student temperatures')
    for value in config['temperatures'].values():
        check(type(value) in (int, float) and math.isfinite(value) and value > 0, 'Invalid student temperature')
    for key in ('max_tokens','max_rows'):
        check(type(config['batching'][key]) is int and config['batching'][key] > 0, 'Invalid batching budget')
    return {'metadata_files_verified':len(files), 'weight_sizes_verified':len(weights), 'alphabet_symbols_verified':256}


def verify_weights(path, config):
    verified = verify_package(path, config)
    for name, pin in config['weights'].items():
        check(sha(local_file(path, name)) == pin['sha256'], 'Student weight SHA256 mismatch: ' + name)
    return {**verified, 'weight_hashes_verified':len(config['weights']),
            'bytes_verified':sum(p['size'] for p in config['weights'].values())}
