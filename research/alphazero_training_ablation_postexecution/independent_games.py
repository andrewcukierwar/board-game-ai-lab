from pathlib import Path
exec(Path(__file__).with_name('independent_review.py').read_text().split('# Independent graph')[0])
out=read(D/'results/independent-recomputed.json')
for label,d in out['independent_training_diagnostics'].items():
 raw=read(E/label/'games.json');extra=read(E/label/'complete.json')['extra'];seen=set();mirrors=set();post=set();postmirrors=set();avoidable=missed=blocked=0
 for g in raw['games']:
  b=board([])
  for p,c in enumerate(g['moves']):
   assert not winner(b,0) and not winner(b,1);assert c in legal(b);a=p%2
   key=(a,tuple(tuple(col) for col in b));mirror=(a,tuple(tuple(col) for col in b[::-1]));canonical=min(key,mirror,key=repr)
   seen.add(key);mirrors.add(canonical)
   if p>=8:post.add(key);postmirrors.add(canonical)
   w=wins(b,a);ss=safe(b,a)
   missed+=bool(w) and c not in w;avoidable+=bool(ss) and c not in ss;blocked+=not w and bool(ss) and len(ss)<len(legal(b)) and c not in ss
   assert all(v==0 for k,v in enumerate(g['visits'][p]) if k not in legal(b));b=play(b,c,a)
  assert g['winner']==(0 if winner(b,0) else 1 if winner(b,1) else -1)
  assert g['winner']!=-1 or len(g['moves'])==42
 n=d['positions'];npst=sum(max(0,len(g['moves'])-8) for g in raw['games'])
 computed={'duplicates':n-len(seen),'mirror_duplicates':n-len(mirrors),'post_ply8_duplicates':npst-len(post),'post_ply8_mirror_duplicates':npst-len(postmirrors),'avoidable_one_reply_errors':avoidable,'missed_immediate_wins':missed,'missed_safe_responses':blocked}
 for k,v in computed.items():assert extra[k]==v,(label,k,v,extra[k])
 d['independent_game_diagnostics']=computed
out['checks']['training_games_board_tactical_diversity_recomputed']=512
(D/'results/independent-recomputed.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('Independent all-game terminal, tactical, duplicate and mirror duplicate checks PASS')
