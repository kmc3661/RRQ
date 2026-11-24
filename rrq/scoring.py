"""User-requested ChartQA formatting extension. Official metric remains separate.
No free-form number extraction, unit deletion, or answer-dependent parsing.
"""
import re,unicodedata
from decimal import Decimal,InvalidOperation
NUM=re.compile(r"^[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]*)?|[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?(?:\s*%)?$")
def parse(text):
 s=unicodedata.normalize('NFKC',str(text)).strip()
 if not NUM.fullmatch(s):return None
 try:
  x=Decimal(s.rstrip('%').strip().replace(',',''))
  return x if x.is_finite() else None
 except InvalidOperation:return None

def judge(prediction,target,official_correct):
 if official_correct:return dict(correct=True,official_correct=True,added=False,reason='official_correct')
 p,t=parse(prediction),parse(target)
 good=p is not None and t is not None and (p==t if t==0 else abs(p-t)<=abs(t)*Decimal('0.05'))
 reason='unchanged_incorrect'
 if good:
  kinds=[]
  if '%' in unicodedata.normalize('NFKC',str(prediction))+unicodedata.normalize('NFKC',str(target)):kinds.append('percent_marker')
  if ',' in str(prediction)+str(target):kinds.append('thousands_comma')
  reason='+'.join(kinds) if kinds else 'numeric_format'
 return dict(correct=bool(good),official_correct=False,added=bool(good),reason=reason)


def mean_retention(scores, fp_scores):
    if set(scores) != set(fp_scores): raise ValueError("Task sets differ")
    if any(v <= 0 for v in fp_scores.values()): raise ValueError("Invalid FP denominator")
    return 100 * sum(scores[k] / fp_scores[k] for k in scores) / len(scores)
