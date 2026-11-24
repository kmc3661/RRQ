"""Generated-token-only, no ground-truth access. Conservative nonperiodic degeneration.
At least32generated tokens. A recent32token window with<=12unique IDs and
>=35%duplicate4gram positions, or4nonoverlapping occurrences of an8gram within64.
"""
def stop_index(tokens):
 for end in range(32,len(tokens)+1):
  w=tokens[end-32:end]
  if len(set(w))<=12:
   grams=[tuple(w[i:i+4]) for i in range(29)]
   if (29-len(set(grams)))/29>=.35:return end
  recent=tokens[max(0,end-64):end];seen={}
  for i in range(len(recent)-7):
   gram=tuple(recent[i:i+8]);positions=seen.setdefault(gram,[])
   if not positions or i-positions[-1]>=8:positions.append(i)
   if len(positions)>=4:return end
 return None
