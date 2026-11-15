"""Portable format-2 schema and content checks; no model construction or downloads."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..questions import option_limits


def check(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def local_file(folder, name):
    check(isinstance(name, str) and name and not PurePosixPath(name).is_absolute()
          and not PureWindowsPath(name).drive and '..' not in PurePosixPath(name).parts
          and '\\' not in name, 'Invalid package file path')
    path = Path(folder) / name
    check(path.absolute().is_relative_to(Path(folder).absolute()), 'Package file escapes its folder')
    check(path.is_file(), 'Missing model file: ' + name)
    return path


def verify_package(path, config):
    from safetensors import safe_open
    from tokenizers import Tokenizer
    check(config.get('quyet_format') == 2 and config.get('kind') == 'adapter', 'Expected format-2 adapter package')
    check(isinstance(config.get('name'), str) and bool(config['name']), 'Missing model name')
    option_limits(config)
    limits = config['limits']
    check(type(limits['max_input_tokens']) is int and limits['max_input_tokens'] > 0
          and limits['truncate_state'] is False, 'Adapter inputs must have a complete-token window without truncation')
    check(config['readouts'] == {'choice':'per_option_head', 'noul':'reference_alphabet', 'score':'reference_alphabet'},
          'Unsupported adapter readout binding')
    check(config['attn_implementation'] == 'sdpa' and config['dtype'] == 'bfloat16', 'Unsupported evaluated compute profile')
    files = config['files']
    required = {config['adapter']['config'], config['adapter']['weights'], config['option_head']['file'],
                'tokenizer.json', 'tokenizer_config.json'}
    check(required <= set(files), 'Incomplete model file hashes')
    for name, expected in files.items():
        check(isinstance(expected,str) and len(expected)==64, 'Invalid SHA256: ' + name)
        check(sha(local_file(path,name)) == expected, 'Model file hash mismatch: ' + name)
    check(config['adapter']['config'] == 'adapter/adapter_config.json'
          and config['adapter']['weights'] == 'adapter/adapter_model.safetensors', 'Unsupported adapter filenames')
    check(config['option_head']['file'].endswith('.safetensors'), 'The option head must use safetensors')
    with safe_open(str(local_file(path,config['option_head']['file'])), framework='pt', device='cpu') as f:
        check(set(f.keys()) == {'weight','bias'}, 'Option head tensor keys differ')
        for key, shape in [('weight',[1,config['option_head']['in_features']]),('bias',[1])]:
            part=f.get_slice(key)
            check(part.get_shape()==shape and part.get_dtype()=='F32', 'Option head tensor shape/dtype differs: '+key)
    base=config['base']
    check(isinstance(base['model_id'],str) and bool(base['model_id']) and isinstance(base['revision'],str)
          and len(base['revision'])==40, 'Missing pinned base identity')
    check(set(base['files'])=={'config.json','model.safetensors.index.json'}, 'Missing pinned base metadata hashes')
    pins=base['shards']
    check(isinstance(pins,dict) and bool(pins), 'Missing base shard pins')
    for name,pin in pins.items():
        check(isinstance(name,str) and name.endswith('.safetensors') and '/' not in name and '\\' not in name, 'Invalid base shard name')
        check(type(pin['size']) is int and pin['size']>0 and isinstance(pin['sha256'],str)
              and len(pin['sha256'])==64 and all(c in '0123456789abcdef' for c in pin['sha256']), 'Invalid base shard pin: '+name)
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
    return {'files_verified':len(files),'alphabet_symbols_verified':256,'head_tensors_verified':2}


def verify_base(path, config):
    path=Path(path)
    check(path.is_dir(), 'Supply a local directory containing the pinned Large base weights')
    for name, expected in config['base']['files'].items():
        check(sha(local_file(path,name))==expected, 'Base metadata hash mismatch: '+name)
    index=json.loads((path/'model.safetensors.index.json').read_text(encoding='utf8'))
    shards=set(index.get('weight_map',{}).values())
    check(bool(shards) and all(s.endswith('.safetensors') for s in shards), 'Invalid safetensors base shard index')
    check(shards==set(config['base']['shards']), 'Base shard index differs from pinned shard set')
    for name in sorted(shards):
        try:shard=local_file(path,name)
        except ValueError as e:
            raise ValueError('Pinned Large base weights are not bundled; missing base shard: '+name) from e
        expected=config['base']['shards'][name]['size'];actual=shard.stat().st_size
        check(actual==expected, f'Base shard size mismatch: {name}; expected {expected} bytes, got {actual}')
    return path


def verify_base_hashes(path, config):
    """Expensive, opt-in full SHA256 verification after the cheap load-time checks."""
    path=verify_base(path,config);verified={};total=0
    for name,pin in config['base']['shards'].items():
        actual=sha(path/name)
        check(actual==pin['sha256'], 'Base shard SHA256 mismatch: '+name)
        verified[name]=actual;total+=pin['size']
    return {'shards_verified':len(verified),'bytes_verified':total,'sha256':verified}
