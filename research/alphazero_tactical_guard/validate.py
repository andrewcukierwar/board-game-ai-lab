"""Bounded synthetic and random legal-position correctness before launch."""
import random
import time
from copy import deepcopy
from collections import Counter
from .common import runtime,load,state,write,RESULTS
from .guard import guard
from .independent_verify import ReferenceBoard,reference_selection,self_check
from .evaluate import decision
from games.connect4.connect4 import Connect4
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.search import EvaluationAgent
from games.connect4.alphazero_v2.oracle import engine_position


def bitboard(ref):return BitboardState.from_game(Connect4([list(r) for r in ref.cells],ref.actor))


def main():
    runtime();start=time.time();self_check();checks=Counter();coverage=Counter();rng=random.Random(4410904)
    def verify(ref):
        root=bitboard(ref);before=deepcopy(root);legal=list(ref.legal())
        c=ref.classify();coverage['both_players_'+str(ref.actor)]+=1
        if c['immediate_winning_actions']:coverage['immediate_win']+=1
        if len(c['immediate_winning_actions'])>1:coverage['multiple_wins']+=1
        if len(c['safe_actions'])>1:coverage['multiple_safe']+=1
        if not c['safe_actions']:coverage['all_unsafe']+=1
        if len(legal)<7:coverage['full_column']+=1
        for original in legal:
            visits=[0]*7;visits[original]=256
            for tied in (False,True):
                vector=[1 if a in legal else 0 for a in range(7)] if tied else visits
                saved=vector.copy();chosen,reason,classification=reference_selection(ref,vector,original)
                result=guard(root,vector,original)
                assert (result['final_action'],result['reason'])==(chosen,reason)
                for key in ('legal_actions','safe_actions','unsafe_actions','opponent_winning_replies'):
                    assert result[key]==classification[key]
                assert result['immediate_wins']==classification['immediate_winning_actions']
                assert result['original_visits']==vector==saved and root==before
                assert chosen in legal and result['selected_visits']==vector[chosen]
                assert result['selected_zero_visits']==(vector[chosen]==0)
                assert result['changed']==(chosen!=original)
                if chosen!=original and vector[chosen]==0:coverage['zero_visit_override']+=1
                if reason=='UNCHANGED':assert chosen==original
                checks['reference_decisions']+=1
    # Physical-column vertical wins, both actors and both reflections.
    designed=[]
    for col in range(7):
        other=(col+1)%7;second=(col+2)%7
        designed.append(ReferenceBoard.from_moves([col,other,col,other,col,second]))
    # Gravity-respecting horizontal and both diagonal fixtures; test color swap as well.
    for stacks in [ ['X','X','X','','O','O','O'],
                    ['X','OX','OOX','XOO','','',''],
                    ['XOO','OOX','OX','X','','',''],
                    ['','X','X','X','','O','O'] ]:
        cells=[[' ']*7 for _ in range(6)]
        for col,stack in enumerate(stacks):
            for height,cell in enumerate(stack):cells[5-height][col]=cell
        designed.append(ReferenceBoard(tuple(tuple(r) for r in cells),0))
    # Bounded legal random positions; stop histories at the first terminal move.
    random_rows=[]
    for _ in range(160):
        ref=ReferenceBoard.empty()
        for ply in range(rng.randrange(1,40)):
            if ref.terminal():break
            ref=ref.play(rng.choice(ref.legal()))
        if ref.terminal():
            try:guard(bitboard(ref),[0]*7,0)
            except ValueError:checks['terminal_rejection']+=1
            else:raise AssertionError('Terminal accepted')
        else:random_rows.append(ref)
    for ref in designed+random_rows:
        for mirror in (False,True):
            cells=tuple(tuple(reversed(r)) if mirror else r for r in ref.cells)
            for swap in (False,True):
                cells2=tuple(tuple({'X':'O','O':'X',' ':' '}[v] for v in r) for r in cells) if swap else cells
                verify(ReferenceBoard(cells2,1-ref.actor if swap else ref.actor))
    # Legal all-unsafe fixture: opposing bottom horizontal double-ended threat.
    cells=[[' ']*7 for _ in range(6)]
    cells[5]=list(' OOO XX')
    verify(ReferenceBoard(tuple(tuple(r) for r in cells),0))
    # Directed terminal draws and actor-relative exact terminal signs.
    draw=[2,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,3,6,4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6]
    from games.connect4.agents.mcts_nn_agent import terminal_value
    verify(ReferenceBoard.from_moves(draw[:-1]))
    assert terminal_value(bitboard(ReferenceBoard.from_moves(draw)))==0
    for ref in designed[:7]:
        for swap in (False,True):
            cells=tuple(tuple({'X':'O','O':'X',' ':' '}[v] for v in r) for r in ref.cells) if swap else ref.cells
            test=ReferenceBoard(cells,1-ref.actor if swap else ref.actor)
            for win in test.classify()['immediate_winning_actions']:
                successor=bitboard(test).drop(win)
                assert successor.winner==test.actor and successor.current_player==1-test.actor
                assert terminal_value(successor)==-1
                try:guard(successor,[0]*7,0)
                except ValueError:checks['directed_terminal_rejection']+=1
                else:raise AssertionError('Winning terminal accepted')
    try:guard(bitboard(ReferenceBoard.from_moves(draw)),[0]*7,0)
    except ValueError:checks['directed_terminal_draw_rejection']+=1
    else:raise AssertionError('Draw terminal accepted')
    assert all(coverage[k]>0 for k in ('multiple_wins','multiple_safe','all_unsafe','full_column','zero_visit_override'))
    # Exact preservation: disabled wrapper versus canonical search and enabled run,
    # including RNG state, model predictions, weights, full learner state, and board.
    for seed in (42,314159):
        runner,model,inference,entry=load(seed);before=state(runner);weights=runner.factory.hash(model)
        for moves in ([],[0,1,0,1,0,2],[3,2,3,2,4,2]):
            game=engine_position(moves);board=deepcopy(game.board);prediction=inference.predict(game)
            r1,r2,r3=[random.Random(4410905) for _ in range(3)]
            canonical,selection=EvaluationAgent(inference,256,rng=r1,backend='bitboard').search(game)
            _,disabled=decision(inference,game,r2,False);_,enabled=decision(inference,game,r3,True)
            assert disabled['original_action']==selection.move==enabled['original_action']
            assert list(canonical.visits)==disabled['visits']==enabled['visits']
            assert disabled['root_value']==enabled['root_value']==canonical.root_value
            assert disabled['root_prior']==enabled['root_prior']==list(canonical.root_prior)
            assert disabled['root_logits']==enabled['root_logits']==list(canonical.root_logits)
            assert r1.getstate()==r2.getstate()==r3.getstate()
            assert prediction==inference.predict(game) and game.board==board
            checks['puct_model_rng_preservation']+=1
        assert before==state(runner) and weights==runner.factory.hash(model)
    write(RESULTS/'validation.json',dict(disposition='PASS',checks=checks,coverage=coverage,
        random_attempts=160,random_nonterminal=len(random_rows),seconds=time.time()-start,
        contracts=['connect4-tactical-root-guard-v1','connect4-guarded-decision-v1'],
        reference='independently authored top-first matrix directional scan'))
    print('Validation PASS')
if __name__=='__main__':main()
