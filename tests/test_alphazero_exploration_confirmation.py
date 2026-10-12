"""Focused contracts for exact confirmation, missingness and decision thresholds."""
import io
import subprocess
from unittest.mock import Mock
import pytest
from research.alphazero_exploration_confirmation import solver, analyze

@pytest.mark.parametrize('before,after,kind',[(1,0,'win_draw'),(1,-1,'win_loss'),(0,-1,'draw_loss'),(1,1,'neutral'),(0,0,'neutral'),(-1,-1,'neutral'),(None,-1,'unresolved'),(1,None,'unresolved'),(None,None,'unresolved')])
def test_exact_classifications(before,after,kind):
    assert solver.classify(before,after)==kind

def test_exact_contradictions_rejected():
    with pytest.raises(ValueError): solver.classify(0,1)
    with pytest.raises(ValueError): analyze.set_value({'a':1},'a',0)

@pytest.mark.parametrize('lo,hi,expected',[(.051,.08,'MATERIAL HARM'),(.05,.08,'INCONCLUSIVE'),(-.02,.049,'NO MATERIAL HARM WITHIN THE DECLARED MARGIN'),(-.02,.05,'INCONCLUSIVE'),(-.1,.1,'INCONCLUSIVE')])
def test_three_way_thresholds(lo,hi,expected):
    assert solver.decision_interval(lo,hi)==expected

def test_timeout_discards_process_and_never_returns_exact(monkeypatch):
    p=Mock(); p.stdin=io.StringIO(); p.stdout=io.StringIO('ok 1 1 1 1 1 1 1 1 0\n')
    session=solver.NativeSession(200000,.3); session.process=p
    monkeypatch.setattr(solver.select,'select',lambda *args:([],[],[]))
    result=session.query([3]*4)
    assert result['status']=='unknown' and result['value'] is None and result['actions']=={}
    assert result['reason']=='wall_timeout' and session.process is None
    p.kill.assert_called_once(); p.communicate.assert_called_once()

def test_node_abort_never_exposes_partial_vector(monkeypatch):
    p=Mock(); p.stdin=io.StringIO(); p.stdout=io.StringIO('unknown 200001\n')
    session=solver.NativeSession(200000,.3); session.process=p
    monkeypatch.setattr(solver.select,'select',lambda *args:([p.stdout],[],[]))
    r=session.query([])
    assert r['status']=='unknown' and r['reason']=='node_limit' and r['actions']=={} and r['value'] is None

def test_illegal_native_result_rejected(monkeypatch):
    p=Mock();p.stdin=io.StringIO();p.stdout=io.StringIO('error invalid_or_terminal 0\n')
    session=solver.NativeSession(1,.3);session.process=p
    monkeypatch.setattr(solver.select,'select',lambda *args:([p.stdout],[],[]))
    with pytest.raises(ValueError): session.query([0]*7)

def game(i,errors,unknown):
    return dict(id=(42,20,0,i),exact=dict(errors=errors,unresolved=unknown))

def test_missingness_retains_full_game_denominator():
    r=[game(0,1,2),game(1,0,0)]
    c=[game(0,2,1),game(1,0,1)]
    b=analyze.effect_bounds(r,c)
    assert b['n']==2 and b['confirmed_difference']==.5
    assert b['identification_interval']==[-.5,1.5]
    assert b['uncertainty_interval'][0]<-.5 and b['uncertainty_interval'][1]>1.5

def test_mismatched_identities_rejected():
    with pytest.raises(ValueError): analyze.effect_bounds([game(0,0,0)],[game(1,0,0)])

@pytest.mark.parametrize('action,tops,kind',[(0,[1,1],'worse_all'),(1,[0,0],'better_all'),(1,[1,1],'equal_all'),(0,[-1,1],'mixed_tie'),(0,[None,0],'unresolved'),(None,[0],'unresolved')])
def test_all_max_visit_ties_explicit(action,tops,kind):
    assert analyze.sampling_comparison(action,tops)['kind']==kind

def test_terminal_win_and_draw_perspective():
    from games.connect4.alphazero_v2.bitboard import BitboardState
    from games.connect4.alphazero_v2.oracle import engine_position
    state=BitboardState.from_game(engine_position([0,1,0,1,0,2]))
    assert analyze.action_value(state,0,{})==1
    assert state.drop(0).current_player==1 and state.drop(0).winner==0
    # 4D archived complete drawn trajectory exercises the last legal draw action.
    import gzip,json
    from pathlib import Path
    for file in sorted(Path('research/alphazero_exploration/results/phase4d4d-20261011/main').glob('*.jsonl.gz')):
        rows=[json.loads(l) for l in gzip.open(file,'rt')]
        draw=next((r for r in rows if r['winner']==-1),None)
        if draw:
            state=BitboardState.from_game(engine_position(draw['moves'][:-1]))
            assert analyze.action_value(state,draw['moves'][-1],{})==0
            break
    else: pytest.fail('Missing previous draw fixture')

def test_reflection_keys_and_action_values():
    from games.connect4.alphazero_v2.bitboard import BitboardState
    from games.connect4.alphazero_v2.oracle import engine_position
    h=[3,2,4,1,4,3,2,0]
    st=BitboardState.from_game(engine_position(h));mir=BitboardState.from_game(engine_position([6-c for c in h]))
    assert analyze.key(st)==analyze.key(mir)
    values={analyze.key(st.drop(1)):-1}
    assert analyze.action_value(st,1,values)==analyze.action_value(mir,5,values)==1

def test_unresolved_coverage_is_not_safe():
    rows=[dict(id=[42],schedule='original',ply=8,stage='8-15',categories=['nonmax_visit'],classification='unresolved',unresolved_reason='node_limit'),
          dict(id=[42],schedule='original',ply=9,stage='8-15',categories=['max_visit'],classification='neutral')]
    c=analyze.coverage(rows)['all']
    assert c==dict(eligible=2,unresolved=1,resolved=1,neutral=1,coverage=.5,pre_value_exact=0,executed_value_exact=0)


def test_incomplete_block_cannot_selectively_keep_one_arm():
    games=[dict(id=(42,20,c,i)) for c in (0,1) for i in range(2)]
    cells=[dict(seed=42,generation=20,cohort=c,schedule=dict(name=n),ids=[0,1],complete=not(c==1 and n=='extended')) for c in (0,1) for n in ('original','extended')]
    kept,excluded=analyze.matched_games(games,cells)
    assert len(kept)==2 and all(g['id'][2]==0 for g in kept)
    assert excluded==[dict(block=(42,20,1),reason='incomplete_or_unmatched_checkpoint_cohort')]


@pytest.mark.parametrize('adequate,complete,cp_upper,expected',[(True,True,.04,'NO MATERIAL HARM WITHIN THE DECLARED MARGIN'),(False,True,.04,'INCONCLUSIVE'),(True,False,.04,'INCONCLUSIVE'),(True,True,.05,'INCONCLUSIVE')])
def test_noninferiority_requires_coverage_both_checkpoints_and_completion(adequate,complete,cp_upper,expected):
    pooled=dict(uncertainty_interval=[-.01,.04]); parts=[dict(uncertainty_interval=[-.02,.02]),dict(uncertainty_interval=[-.01,cp_upper])]
    assert analyze.safety_conclusion(pooled,parts,adequate,complete)==expected

def test_material_harm_needs_complete_design_but_not_full_coverage():
    pooled=dict(uncertainty_interval=[.06,1.0]); parts=[pooled,pooled]
    assert analyze.safety_conclusion(pooled,parts,False,True)=='MATERIAL HARM'
    assert analyze.safety_conclusion(pooled,parts,True,False)=='INCONCLUSIVE'


def old_decisive_game():
    import gzip,json
    from pathlib import Path
    from games.connect4.alphazero_v2.oracle import engine_position
    from games.connect4.alphazero_v2.bitboard import BitboardState
    for file in sorted(Path('research/alphazero_exploration/results/phase4d4d-20261011/main').glob('*original.jsonl.gz')):
        raw=next((json.loads(line) for line in gzip.open(file,'rt') if len(json.loads(line)['moves'])>8 and json.loads(line)['winner']!=-1),None)
        if raw:
            state=BitboardState.from_game(engine_position([])); rows=[]
            for ply,action in enumerate(raw['moves']):
                rows.append(dict(ply=ply,action=action,top={action},wins={c for c in state.get_valid_moves() if state.drop(c).winner==state.current_player},threats=[]))
                state=state.drop(action)
            return dict(id=(42,20,0,0),moves=raw['moves'],schedule='extended',plies=rows)
    pytest.fail('Missing old decisive fixture')


def test_terminal_executed_win_proves_neutral_with_missing_pre_value():
    g=old_decisive_game()
    rows=analyze.adjudicate([g],dict(values={},queries=[]))
    last=rows[-1]
    assert last['classification']=='neutral' and last['before']==1 and last['action_value']==1
    assert g['exact']['unresolved']>0 and g['exact'].get('errors',0)==0


def test_impossible_lost_scalar_on_winning_action_is_rejected():
    from games.connect4.alphazero_v2.oracle import engine_position
    from games.connect4.alphazero_v2.bitboard import BitboardState
    g=old_decisive_game(); state=BitboardState.from_game(engine_position(g['moves'][:-1]))
    with pytest.raises(ValueError): analyze.adjudicate([g],dict(values={analyze.key(state):-1},queries=[]))
