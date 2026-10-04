"""Portable replay builds use measured data without importing model libraries."""
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
EXAMPLE=ROOT/'examples/gem_run'


def replay_module():
    path=EXAMPLE/'replay.py'
    assert path.exists(), 'The release example needs a standalone replay builder'
    spec=importlib.util.spec_from_file_location('gem_replay_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_loader_uses_only_standard_library():
    replay_module()
    run=subprocess.run([sys.executable,'-S',str(EXAMPLE/'replay.py'),'--help'],cwd=ROOT,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert '--records' in run.stdout


def test_sample_is_only_the_requested_pairs_and_stops_on_first_rock():
    data=json.loads(gzip.decompress((EXAMPLE/'recordings/sample.json.gz').read_bytes()))
    assert data['pairings']=={'en':['small_en','laya_english'],'vi':['small','laya_multilingual']}
    assert data['provenance']['kind']=='recorded_survival_prefix'
    for model,languages in data['races'].items():
        for language,seeds in languages.items():
            assert model in data['pairings'][language]
            for seed,race in seeds.items():
                course=data['courses'][seed]
                assert sum(r['collision'] for r in race['results'])==1
                assert race['results'][-1]['collision']
                assert race['duration']==course[race['results'][-1]['row_id']]['end']
                assert all(c['received']<=race['duration'] for c in race['calls'])
    assert (EXAMPLE/'recordings/sample.json.gz').stat().st_size<500_000
    text=json.dumps(data)
    for marker in ('/ho'+'me/','/ro'+'ot/','/t'+'mp/','/work'+'space/'):
        assert marker not in text


def test_live_records_allow_one_model_custom_seed_and_course(tmp_path):
    m=replay_module()
    data=json.loads(gzip.decompress((EXAMPLE/'recordings/sample.json.gz').read_bytes()))
    race=data['races']['small_en']['en']['701']
    race={**race,'seed':913,'course':data['courses']['701']}
    (tmp_path/'small_en_race_en_913.json').write_text(json.dumps(race))
    bundled=m.load_records(tmp_path)
    assert bundled['models']==['small_en']
    assert bundled['pairings']=={'en':['small_en']}
    assert set(bundled['courses'])=={'913'}
    assert bundled['provenance']['kind']=='live_run'


def test_different_courses_cannot_be_compared_as_the_same_seed(tmp_path):
    m=replay_module()
    data=json.loads(gzip.decompress((EXAMPLE/'recordings/sample.json.gz').read_bytes()))
    for model in ['small_en','laya_english']:
        race=data['races'][model]['en']['701']
        race={**race,'course':json.loads(json.dumps(data['courses']['701']))}
        if model=='laya_english':race['course'][0]['goal']='yellow'
        (tmp_path/f'{model}_race_en_701.json').write_text(json.dumps(race))
    with pytest.raises(ValueError,match='course'):
        m.load_records(tmp_path)


def test_builder_escapes_script_content_and_embeds_background(tmp_path):
    m=replay_module()
    template=tmp_path/'template.html'
    template.write_text('<script type="application/json">__GEM_DATA__</script><img src="__GEM_BACKGROUND__">')
    image=tmp_path/'world.png'
    image.write_bytes(b'example image bytes')
    # Runtime metadata is an untrusted string and cannot close the data script.
    data=m.load_records(None)
    data['runtime']['small_en']['display_name']='</script><script>alert(1)</script>'
    out=tmp_path/'replay.html'
    m.write_replay(data,out,template=template,background=image)
    rendered=out.read_text()
    assert '<script>alert(1)</script>' not in rendered
    assert '\\u003c/script\\u003e' in rendered
    assert 'data:image/png;base64,' in rendered
    assert '__GEM_' not in rendered


def test_empty_record_directory_has_actionable_error(tmp_path):
    with pytest.raises(ValueError,match='No race records'):
        replay_module().load_records(tmp_path)


@pytest.mark.parametrize('seed', [9007199254740992, -9007199254740992])
def test_record_seeds_must_survive_browser_json_parsing(tmp_path, seed):
    data=replay_module().load_records(None)
    race={**data['races']['small_en']['en']['701'], 'seed':seed,
          'course':data['courses']['701']}
    (tmp_path/f'small_en_race_en_{seed}.json').write_text(json.dumps(race))
    with pytest.raises(ValueError,match='seed.*9007199254740991'):
        replay_module().load_records(tmp_path)
