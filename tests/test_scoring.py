from rrq.scoring import judge, mean_retention
from rrq.stopping_reference import stop_index

def test_chartqa_format_extension_does_not_extract_numbers():
    assert judge('1,408','1408',False)['correct']
    assert judge('2%','2',False)['correct']
    assert not judge('2 people','2',False)['correct']
    assert not judge('0.02','2%',False)['correct']
    assert judge('existing official match','different',True)['correct']

def test_retention_is_task_ratio_mean():
    assert mean_retention({'a':20,'b':90},{'a':10,'b':100}) == 145

def test_stop_uses_generated_ids_only():
    assert stop_index([1]*31) is None
    assert stop_index([1]*32) == 32
    assert stop_index(list(range(100))) is None
