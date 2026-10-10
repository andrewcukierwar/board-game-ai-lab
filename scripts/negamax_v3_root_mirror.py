"""Research-only symmetry policy chosen once from exact initial-root geometry."""
from random import Random
from scripts.negamax_v3_variants import variant_source,replace,load_source
from scripts.benchmark_public_agents import position


def source_for(baseline):
    source=variant_source('mirror-selective',baseline=baseline)
    source=replace(source,'        self.entries = {}',
                   '        self.mirror_enabled = False\n        self.entries = {}')
    source=replace(source,'    key, mirrored = identity(state, depth, 3)',
                   '    mirrored = False\n'
                   '    if table is not None and table.mirror_enabled:\n'
                   '        key, mirrored = identity(state, depth, 3)\n'
                   '    else:\n'
                   '        key = (state.pieces[0] | (state.pieces[1] << 49) |\n'
                   '               (state.mover << 98) | (depth << 99))')
    source=replace(source,'        table, scores = SearchTable(), {}',
                   '        table, scores = SearchTable(), {}\n'
                   '        table.mirror_enabled = (reflect(state.pieces[0]) == state.pieces[0] and\n'
                   '                                reflect(state.pieces[1]) == state.pieces[1])')
    return source


def symmetric_fixtures(existing,baseline,seed=20261013):
    rng=Random(seed)
    module=load_source(source_for(baseline))
    def identity(history):
        s=module.SearchState(position(history))
        return (*s.pieces,s.mover)
    seen={identity(p['history']) for p in existing}
    result=[]
    for i,plies in enumerate((4,8,9,16,17,25)):
        for attempt in range(100000):
            history=[]
            game=position([])
            valid=True
            for _ in range(plies//4):
                a,b=rng.randrange(4),rng.randrange(4)
                for col in (a,b,6-a,6-b):
                    if col not in game.get_valid_moves() or game.is_game_over():
                        valid=False
                        break
                    assert game.make_move(col)
                    history.append(col)
                if not valid:break
            if valid and plies%4:
                if 3 not in game.get_valid_moves() or game.is_game_over():valid=False
                else:
                    game.make_move(3)
                    history.append(3)
            if not valid or game.is_game_over():continue
            state=module.SearchState(game)
            key=(*state.pieces,state.mover)
            assert len(history)==plies
            assert all(module.reflect(bits)==bits for bits in state.pieces)
            if key in seen:continue
            seen.add(key)
            result.append(dict(id=f'symmetric-{i:02d}',history=history,group='symmetric-target'))
            break
        else:raise RuntimeError('Symmetric legal fixture generation budget exhausted')
    return result
