import json
from rrq.evaluation import read_manifest, generation_options

def test_sampling_settings_preserved():
    row={'generation':{'max_new_tokens':128,'do_sample':True,'temperature':0.7,'top_p':0.8}}
    original=dict(row['generation']);actual=generation_options(row)
    assert actual['do_sample'] is True and actual['temperature']==0.7 and actual['top_p']==0.8
    assert row['generation']==original

def test_manifest_unicode_separator_is_not_a_new_record(tmp_path):
    row={'id':'a','task':'chartqa','images':['chart.png'],'prompt':'before\u2028after','generation':{'max_new_tokens':16}}
    path=tmp_path/'manifest.jsonl';path.write_text(json.dumps(row,ensure_ascii=False)+'\n')
    assert read_manifest(path)==[row]

def test_manifest_rejects_duplicate_ids(tmp_path):
    row={'id':'a','task':'ai2d','images':['a.png'],'prompt':'Q','generation':{'max_new_tokens':16}}
    path=tmp_path/'manifest.jsonl';path.write_text((json.dumps(row)+'\n')*2)
    try:read_manifest(path)
    except ValueError as error:assert 'Duplicate' in str(error)
    else:raise AssertionError('duplicate accepted')
