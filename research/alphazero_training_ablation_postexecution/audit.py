"""Strict evidence audit. Never calls model inference, search or training."""
from collections import Counter
from copy import deepcopy
import math
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from research.alphazero_training_ablation.common import ROOT, REPO, OUTPUT, HISTORICAL, read, write, sha, enforce, preserve
from research.alphazero_training_ablation.manage import ORDER
from research.alphazero_training_ablation.worker import state, extra
from games.connect4.alphazero_v2.experimental import load_experimental, game_record, digest, ORIGINAL, EXTENDED
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime
from games.connect4.alphazero_v2.batching import game_seed
from games.connect4.alphazero_v2.oracle import engine_position

PROSPECTIVE='b4b1d9f5a42db0889fb2eed2acea6252da8526c9'
RESULTS=ROOT/'results'

def require(condition, message):
    if not condition: raise ValueError(message)

def computational(s):
    result=deepcopy(s)
    result['source']=result['source']['execution']
    return result

def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()

def protected():
    paths=git(REPO,'ls-tree','-r','--name-only',PROSPECTIVE).splitlines()
    # Everything predating this phase, plus historical/ResNet worktrees.
    records={p:sha(REPO/p) for p in paths if p!='docs/phase4d4g-alphazero-training-ablation.md'
             and not p.startswith('research/alphazero_training_ablation/') and p!='tests/test_alphazero_ablation.py'}
    worktrees={}
    for name,path in [('historical',HISTORICAL),('resnet',HISTORICAL/'experiment-output/phase4d4h-worktree'),
                      ('resnet-b',HISTORICAL/'experiment-output/phase4d4h-b-worktree')]:
        worktrees[name]=dict(path=str(path),head=git(path,'rev-parse','HEAD'),status=git(path,'status','--porcelain'),
            files={p:sha(path/p) for p in git(path,'ls-files').splitlines() if (path/p).is_file()})
    return dict(prior_tracked_files=records,worktrees=worktrees)

def protected_content(record):
    """Tracked bytes and HEAD are protection gates; untracked status is reporting."""
    result=deepcopy(record)
    for worktree in result['worktrees'].values(): worktree.pop('status')
    return result

def structure():
    status=read(OUTPUT/'execution/status.json'); fixtures=read(ROOT/'fixtures.json')
    require(status['status']=='COMPLETE_PENDING_FINAL_ANALYSIS','Incomplete manager status')
    require(status['design_commit']==PROSPECTIVE,'Wrong prospective commit')
    require(status['order']==[list(x) for x in ORDER],'Wrong training order')
    expected=['parent-42','parent-314159']
    for seed,arm,g in ORDER: expected += [f'{seed}-{arm}-g{g}',f'{seed}-{arm}-g{g}-evaluation']
    expected += ['42-control-strength','42-treatment-strength','314159-treatment-strength','314159-control-strength']
    require([t['label'] for t in status['tasks']]==expected,'Missing, repeated or reordered task')
    actual=[p.name for p in (OUTPUT/'execution').iterdir() if p.is_dir()]
    require(set(actual)==set(expected),'Unreported attempt directory')
    require(not list(OUTPUT.rglob('*.tmp')),'Unpublished temporary archive')
    require(status['elapsed_seconds']<10800,'Total deadline exceeded')
    require(status['primary_games']==status['primary_updates']==512,'Incorrect totals')
    hashes={}; roots=0; match_moves=0; simulation_games=0; completed_counts=Counter()
    for task in status['tasks']:
        directory=Path(task['output'])
        require(task['state']=='COMPLETE','Incomplete task')
        require(read(directory/'task-complete.json')==task['resources'],'Task sidecar differs')
        require(task['resources']['complete'],'False task completion')
        require(task['resources']['wall_seconds']<task['limit_seconds'],'Task deadline exceeded')
        require(not Path(task['log']).read_text().strip(),'Nonempty worker failure log')
        if task['task']=='train':
            summary=read(directory/'complete.json')
            require(summary['complete'] and summary['result']['games']==64 and summary['result']['updates']==64,'Wrong generation counts')
            require(summary['events']['game_finalized']==summary['events']['after_update']==64,'Wrong completed operation count')
            require(sha(summary['boundary'])==summary['sha256']==summary['result']['boundary_sha256'],'Wrong boundary SHA')
            completed_counts['games']+=64; completed_counts['updates']+=64
            simulation_games += summary['result']['new_positions']*256
        elif task['task']=='positions':
            for suite in ('solved','tactical','deep'):
                data=read(directory/f'{suite}.json')
                require(data['complete'] and len(data['rows'])==len(fixtures[suite]),'Incomplete position evaluation')
                require([r['id'] for r in data['rows']]==[r['id'] for r in fixtures[suite]],'Evaluation membership/order differs')
                for row,label in zip(data['rows'],fixtures[suite]):
                    legal=engine_position(label['moves']).get_valid_moves()
                    require(row['cluster']==label['cluster'] and row['family']==label['family'],'Cluster differs')
                    require(row['policy_action'] in legal and row['puct_action'] in legal,'Illegal evaluation action')
                    require(sum(row['visits'])==512 and row['visits'][row['puct_action']]==max(row['visits']),'Wrong PUCT action/budget')
                    require(all(row['visits'][a]==0 and row['policy'][a]==0 for a in range(7) if a not in legal),'Illegal action mass')
                    require(abs(sum(row['policy'])-1)<1e-9 and all(math.isfinite(p) and p>=0 for p in row['policy']),'Invalid policy')
                    require(row['policy'][row['policy_action']]==max(row['policy']),'Wrong raw policy argmax')
                    if suite=='tactical':
                        for method in ('policy','puct'):
                            action=row[f'{method}_action']
                            require(row[f'{method}_correct']==int(action==label['expected_action']),'Wrong tactical label')
                            require(row[f'{method}_avoidable']==int(action not in label['safe_actions']),'Wrong safe label')
                            require(row[f'{method}_missed_win']==int(bool(label['immediate_wins']) and action not in label['immediate_wins']),'Wrong win label')
                    else:
                        require(set(map(int,label['action_values']))==set(legal),'Incomplete oracle vector')
                        mass=sum(row['policy'][a] for a in label['optimal_actions'])
                        require(abs(row['ce']+math.log(max(mass,1e-300)))<1e-12 and abs(row['optimal_mass']-mass)<1e-12,'Wrong CE/mass')
                        require(row['accuracy']==int(row['policy_action'] in label['optimal_actions']),'Wrong optimal accuracy')
                        require(abs(row['mse']-(row['value']-label['value'])**2)<1e-12,'Wrong value MSE')
                        for method in ('policy','puct'):
                            require(row[f'{method}_deep_mistake']==int(label['action_values'][str(row[f'{method}_action'])]<label['value']),'Wrong deep label')
                roots+=len(data['rows'])
        else:
            data=read(directory/'matches.json'); require(data['complete'] and len(data['records'])==64,'Incomplete strength matches')
            pairs=[(o['id'],c) for o in fixtures['openings'] for c in (0,1)]
            require([(r['opening_id'],r['agent_color']) for r in data['records']]==pairs,'Wrong opening/color pairing')
            for record in data['records']:
                opening=next(o for o in fixtures['openings'] if o['id']==record['opening_id'])
                require(not record['abandoned'] and record['moves'][:len(opening['moves'])]==opening['moves'],'Wrong opening/abandoned game')
                game=engine_position(record['moves'])
                require(game.is_game_over() and game.check_winner()==record['winner'],'Incomplete/wrong winner')
                require(record['result']==('draw' if record['winner']==-1 else 'win' if record['winner']==record['agent_color'] else 'loss'),'Wrong color-relative result')
                require(len(record['decisions'])==len(record['moves'])-len(opening['moves']),'Decision/history count differs')
                for i,d in enumerate(record['decisions']):
                    ply=len(opening['moves'])+i
                    require(d['color']==ply%2 and d['move']==record['moves'][ply],'Wrong decision actor')
                    require(sum(d['visits'])==512 and d['visits'][d['move']]==max(d['visits']),'Wrong match search budget/action')
                match_moves+=len(record['decisions'])
                completed_counts['matches']+=1
        for p in sorted(directory.glob('*.json')):
            if not p.name.endswith('-partial.json'): hashes[str(p.relative_to(OUTPUT))]=sha(p)
    require(roots==6280 and completed_counts==Counter(games=512,updates=512,matches=256),'Completed totals differ')
    return dict(status=status,task_count=len(expected),counts=dict(completed_counts),evaluation_search_roots=roots,
        selfplay_simulations=simulation_games,position_evaluation_simulations=roots*512,
        strength_decisions=match_moves,strength_simulations=match_moves*512,evidence_hashes=hashes,
        retry_evidence='Exact task/directory/order equality, all expected success markers, empty failure logs, no tmp archives; no observed retry')

def boundaries():
    readiness=read(ROOT/'results/readiness.json'); records={}; parent_state={}; sim=0
    for seed in (42,314159):
        for arm in ('control','treatment'):
            source=readiness['seeds'][str(seed)]['arms'][arm]
            previous=None
            for generation in (20,21,22):
                completion=None
                if generation>20:
                    completion=read(OUTPUT/f'execution/{seed}-{arm}-g{generation}/complete.json')
                    source=dict(path=completion['boundary'],sha256=completion['sha256'])
                runner=load_experimental(source['path'],expected_sha256=source['sha256'],restore_global_rng=False)
                computed=state(runner); computational_hash=runner.initial_state_sha256()
                require(runner.completed_generations==generation and runner.arm==arm,'Wrong generation/arm')
                require(runner.schedule==(ORIGINAL if arm=='control' else EXTENDED),'Wrong schedule')
                require(runner.config.training.seed==seed and runner.config.training.games_per_generation==64 and
                    runner.config.training.self_play_simulations==256 and runner.config.optimizer_updates==64 and
                    runner.config.training.batch_size==128 and runner.config.pool_size==8 and runner.config.training.max_generations==22,'Wrong config')
                require(runner.parent['file_sha256']==readiness['seeds'][str(seed)]['parent_sha256'],'Wrong historical parent')
                expected_state=readiness['seeds'][str(seed)]['arms'][arm]['state'] if generation==20 else completion['after']
                require(computational(computed)==computational(expected_state),'Deterministic state differs')
                require(runner.replay.generations==tuple(range(generation-7,generation+1)),'Replay expiration differs')
                if generation==20:
                    require(computational_hash==readiness['seeds'][str(seed)]['arms'][arm]['initial_state_sha256'],'Initial state hash differs')
                    if seed in parent_state: require(parent_state[seed]==computational(computed),'Initial arms differ')
                    else: parent_state[seed]=computational(computed)
                else:
                    require(computational(previous)==computational(completion['before']),'Lineage input state differs')
                    games=[g for g in runner.replay.iter_games() if g.generation==generation]
                    require(len(games)==64 and [g.index for g in games]==list(range(64)),'Wrong complete-game identities')
                    stored=read(OUTPUT/f'execution/{seed}-{arm}-g{generation}/games.json')
                    require(stored['games']==[game_record(g) for g in games] and stored['metadata']==runner.metadata[generation],'Exported game records differ')
                    # JSON sidecars stringify integer histogram keys and turn tuples into arrays.
                    expected_history=json.loads(json.dumps(runner.history[-1]))
                    require(completion['result']==expected_history | {'boundary_sha256':source['sha256'],'resources':completion['result']['resources']},'Deterministic history differs')
                    require(completion['extra']==__import__('json').loads(__import__('json').dumps(extra(runner))),'Diagnostics differ')
                    expected_rng=[dict(index=i,seeds={domain:game_seed(seed,generation,i,domain) for domain in ('search','action','noise')}) for i in range(64)]
                    require(completion['game_rng_identities']==expected_rng,'Per-game RNG identities differ')
                    for g in games:
                        for e in g.examples:
                            require(sum(e.visits)==256 and all(abs(p-v/256)<1e-12 for p,v in zip(e.policy_target,e.visits)),'Invalid target/budget')
                            require(e.outcome==(0 if g.winner==-1 else 1 if e.actor==g.winner else -1),'Invalid actor-relative label')
                    sim+=sum(len(g.moves) for g in games)*256
                records[f'{seed}-{arm}-g{generation}']=dict(path=source['path'],file_sha256=sha(source['path']),
                    computational_state_sha256=computational_hash,state=computed,
                    boundary_integrity=torch.load(source['path'],map_location='cpu',weights_only=True)['integrity'],replay_counts=runner.replay.counts(),
                    model_collection_immutability='Frozen execution checks both snapshot/learner before replay; model hash chain verified' if generation>20 else 'parent',
                    atomic_publication='Frozen fsync/strict-reload/no-replace-link/directory-fsync path; successful task and no temp remnants',
                    source_runtime_strict_reload=True)
                previous=computed
    return dict(records=records,selfplay_simulations=sim,
        evaluation_checkpoint_assignment='Reconstructed from frozen manager current-boundary transitions/order and strict evaluate_boundary loader; no per-row checkpoint attestation and no repeated neural queries')

def main():
    started=time.monotonic(); configure_deterministic_runtime(1)
    enforce(); official=preserve(); baseline=protected()
    existing=RESULTS/'protected-before.json'
    if existing.exists(): require(protected_content(read(existing))==protected_content(baseline),'Protected files changed since audit start')
    else: write(existing,baseline)
    result=dict(structure=structure(),boundaries=boundaries(),official_files=official,
        prospective_commit=PROSPECTIVE,freeze_sha256=sha(ROOT/'results/freeze.json'),fixture_sha256=sha(ROOT/'fixtures.json'))
    require(result['structure']['selfplay_simulations']==result['boundaries']['selfplay_simulations'],'Simulation accounting differs')
    after=protected(); write(RESULTS/'protected-after.json',after)
    require(preserve()==official and protected_content(after)==protected_content(baseline),'Protected evidence changed during audit')
    result['protected_reporting']={name:dict(before=read(existing)['worktrees'][name]['status'],after=after['worktrees'][name]['status'])
        for name in after['worktrees'] if read(existing)['worktrees'][name]['status']!=after['worktrees'][name]['status']}
    enforce(); result.update(status='PASS',wall_seconds=time.monotonic()-started,
        caveats=['Git reporting changes are excluded from computational equality; execution-content/runtime remain strictly enforced.',
                 'Atomic syscall provenance follows frozen reviewed source and successful completion; not an independent syscall trace.',
                 'No inference/search/training rerun; evaluation identities follow the frozen orchestrator, not per-row signatures.'])
    write(RESULTS/'integrity-audit.json',result)
    print(dict(status=result['status'],seconds=result['wall_seconds'],counts=result['structure']['counts']))

if __name__=='__main__': main()
