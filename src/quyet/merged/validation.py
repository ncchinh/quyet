"""Local merged checkpoint integrity, including Hub snapshot symlinks."""
import json
import math
from pathlib import Path
from ..adapter.validation import check, local_file, sha
from ..questions import option_limits


def verify_package(path, config):
    from safetensors import safe_open
    from tokenizers import Tokenizer
    check(config.get('quyet_format') == 2 and config.get('kind') == 'merged', 'Expected format-2 merged package')
    check(isinstance(config.get('name'), str) and bool(config['name']), 'Missing model name')
    option_limits(config)
    limits = config['limits']
    check(type(limits['max_input_tokens']) is int and limits['max_input_tokens'] > 0
          and limits['truncate_state'] is False, 'Merged inputs must have a complete-token window without truncation')
    check(config['readouts'] == {'choice':'per_option_head', 'noul':'reference_alphabet', 'score':'reference_alphabet'},
          'Unsupported merged readout binding')
    check(config['attn_implementation'] == 'sdpa' and config['dtype'] == 'bfloat16', 'Unsupported evaluated compute profile')
    path = Path(path)
    files, weights = config['files'], config['weights']
    required = {'config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json', config['option_head']['file']}
    check(set(files) == required, 'Incomplete merged metadata hashes')
    check(isinstance(weights, dict) and bool(weights), 'Missing merged weight pins')
    expected = set(files) | set(weights) | {'quyet_config.json'}
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}
    check('adapter_config.json' not in actual, 'Merged model must not contain an adapter_config.json that could trigger adapter auto-loading')
    unexpected = {name for name in actual - expected if Path(name).suffix.lower()
                  in {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf', '.onnx'}}
    check(expected <= actual and not unexpected, 'Merged model file set differs: missing=' + str(sorted(expected-actual)) + ', unexpected=' + str(sorted(unexpected)))
    for name, pin in {**{k:{'sha256':v} for k,v in files.items()}, **weights}.items():
        digest = pin['sha256']
        check(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Invalid SHA256: ' + name)
        p = local_file(path, name)
        if name in files:
            check(sha(p) == digest, 'Merged metadata hash mismatch: ' + name)
        else:
            check('/' not in name and name.endswith('.safetensors'), 'Invalid merged weight filename')
            check(type(pin['size']) is int and pin['size'] > 0, 'Invalid merged weight size: ' + name)
            check(p.stat().st_size == pin['size'], 'Merged weight size mismatch: ' + name)
    index = json.loads((path/'model.safetensors.index.json').read_text(encoding='utf8'))
    weight_map = index.get('weight_map', {})
    check(bool(weight_map) and set(weight_map.values()) == set(weights), 'Merged shard index differs from pinned weight set')
    seen = set()
    for name in weights:
        with safe_open(str(local_file(path, name)), framework='pt', device='cpu') as f:
            for key in f.keys():
                check(key not in seen and weight_map.get(key) == name, 'Unexpected, duplicate or misplaced merged tensor: ' + key)
                seen.add(key)
    check(seen == set(weight_map), 'Missing merged tensors: ' + str(sorted(set(weight_map)-seen)))
    check(config['option_head']['file'].endswith('.safetensors'), 'The option head must use safetensors')
    with safe_open(str(local_file(path,config['option_head']['file'])), framework='pt', device='cpu') as f:
        check(set(f.keys()) == {'weight','bias'}, 'Option head tensor keys differ')
        for key, shape in [('weight',[1,config['option_head']['in_features']]),('bias',[1])]:
            part=f.get_slice(key)
            check(part.get_shape()==shape and part.get_dtype()=='F32', 'Option head tensor shape/dtype differs: '+key)
    binding=config['renderer']
    check(binding['version']=='gemma4-w8-diet-d-v1' and binding['mode']=='per_option_head', 'Unknown renderer version')
    check(binding['delimiter_token']=='<unused0>', 'Wrong delimiter token')
    tok=Tokenizer.from_file(str(local_file(path,'tokenizer.json')))
    tok.no_truncation();tok.no_padding();tok.encode_special_tokens=False
    check(tok.token_to_id(binding['delimiter_token'])==binding['delimiter_id'], 'Tokenizer delimiter binding differs')
    check(binding['symbols']==list('ABCDEFGHIJ') and len(binding['symbol_ids'])==10, 'Missing ten-symbol letter binding')
    check(len(binding['reference_symbols'])==len(binding['reference_symbol_ids'])==256
          and len(set(binding['reference_symbols']))==256 and len(set(binding['reference_symbol_ids']))==256,
          'Missing 256-symbol alphabet binding')
    check(binding['reference_symbols'][:10]==binding['symbols']
          and binding['reference_symbol_ids'][:10]==binding['symbol_ids'], 'Alphabet/letter prefix differs')
    for symbol, token in zip(binding['reference_symbols'],binding['reference_symbol_ids']):
        check(type(token) is int and token>=0 and tok.encode(symbol,add_special_tokens=False).ids==[token],
              'Tokenizer alphabet binding differs: '+symbol)
    for field in ('prefix_ids','suffix_ids'):
        check(isinstance(binding[field],list) and bool(binding[field])
              and all(type(i) is int and i>=0 and tok.id_to_token(i) is not None for i in binding[field]),
              'Invalid chat affix binding: '+field)
    for value in config.get('temperatures',{}).values():
        check(isinstance(value,(int,float)) and math.isfinite(value) and value>0, 'Invalid fitted temperature')
    for key in ('max_tokens','max_rows'):
        check(type(config['batching'][key]) is int and config['batching'][key]>0, 'Invalid batching budget')
    return {'metadata_files_verified':len(files), 'weight_sizes_verified':len(weights), 'tensors_verified':len(seen), 'alphabet_symbols_verified':256, 'head_tensors_verified':2}



def verify_weights(path, config):
    """Opt-in full shard hashes; ordinary loads check metadata, sizes and tensor inventory."""
    verified = verify_package(path, config)
    for name, pin in config['weights'].items():
        check(sha(local_file(path, name)) == pin['sha256'], 'Merged weight SHA256 mismatch: ' + name)
    return {**verified, 'weight_hashes_verified':len(config['weights']),
            'bytes_verified':sum(pin['size'] for pin in config['weights'].values())}
