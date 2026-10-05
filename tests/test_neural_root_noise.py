"""Root-noise contracts with fake policies; no learned-strength assertions."""
from copy import deepcopy
from dataclasses import asdict
import json
import random

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from games.connect4.agents.mcts_nn_agent import MCTSNNAgent, RootDirichletNoise, visit_policy
from games.connect4.neural_mcts import legal_policy
from games.connect4 import neural_self_play as driver
from games.connect4.neural_evaluation import evaluate, snapshot
from games.connect4.train_mcts_nn import capture_example, finalize_examples, reflect_completed_example
from test_connect4_neural_mcts import FixedLogits, position, DRAW, bounded_torch_threads


def agent(**kwargs):
    return MCTSNNAgent(FixedLogits([0., 1., 2., 3., 4., 5., 6.]), 32,
                       rng=random.Random(42), **kwargs)


def tree_stats(node):
    return (node.visits, node.value_sum, node.prior_p,
            {a: tree_stats(c) for a,c in node.children.items()})


@pytest.mark.parametrize('moves', [[], [0]*6, [0,1,0,1,0,2], DRAW[:-1]])
def test_zero_epsilon_exact_original_priors_search_no_draws(moves):
    game = position(moves)
    original, disabled = agent(), agent(root_noise_epsilon=0., root_dirichlet_alpha=9., root_noise_seed=71)
    before = disabled.root_noise.rng_state()
    # Original path: root's normal legal priors are returned verbatim.
    original.root_noise.mix = lambda priors, moves: (priors, ())
    a,b = original.search(game), disabled.search(game)
    assert (a.move,a.policy,a.visits) == (b.move,b.policy,b.visits)
    assert tree_stats(a.root) == tree_stats(b.root)
    assert b.root_prior == b.root_prior_before_noise
    assert np.array_equal(b.root_prior, legal_policy(disabled.inference.predict_legal(game).policy,
                                                   game.get_valid_moves()))
    assert disabled.root_noise.rng_state() == before and disabled.root_noise.draws == 0
    assert b.root_noise_sample == () and b.root_noise_draw_index is None


@pytest.mark.parametrize('moves,guard', [([],False), ([0]*6,False), (DRAW[:-1],False),
                                       ([0,1,0,1,0,2],True)])
def test_legal_dimensionality_exact_mixing_normalization_root_only(moves,guard):
    search = agent(root_noise_epsilon=.25, root_dirichlet_alpha=.30, tactical_guard=guard)
    game = position(moves)
    before = deepcopy(game.__dict__)
    prediction = search.inference.predict(game)
    result = search.search(game)
    permitted = sorted(result.root.children)
    oracle = np.random.Generator(np.random.PCG64(int(search.root_noise.seed_digest,16)))
    noise = np.zeros(7)
    noise[permitted] = oracle.dirichlet(np.full(len(permitted),.30))
    expected = .75*np.array(result.root_prior_before_noise)+.25*noise
    expected /= expected.sum()
    assert np.array_equal(noise,result.root_noise_sample)
    assert np.array_equal(expected,result.root_prior)
    assert sum(result.root_prior) == pytest.approx(1.)
    assert all(np.isfinite(p) and p >= 0 for p in result.root_prior)
    assert all(result.root_prior[a] == 0 for a in range(7) if a not in permitted)
    assert search.root_noise.draws == result.root_noise_draw_index == 1
    assert game.__dict__ == before and search.inference.predict(game) == prediction
    assert result.root.visits == sum(result.visits) == 32
    assert np.array_equal(result.policy, visit_policy(result.visits,result.temperature))
    assert capture_example(result).policy == result.policy
    def check(node):
        if node.children:
            priors = legal_policy(search.inference.predict_legal(node.game_state).policy,
                                  node.game_state.get_valid_moves())
            assert all(child.prior_p == priors[a] for a,child in node.children.items())
            for child in node.children.values():
                check(child)
    for child in result.root.children.values():
        check(child)


@pytest.mark.parametrize('epsilon', [-.1,1.1,float('nan'),float('inf'),True,'0.25'])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        RootDirichletNoise(epsilon)


@pytest.mark.parametrize('alpha', [0,-.1,float('nan'),float('inf'),True,'0.3'])
def test_invalid_alpha_even_disabled(alpha):
    with pytest.raises(ValueError):
        RootDirichletNoise(0.,alpha)


def test_domain_rng_ownership_json_state_replay():
    a,b = agent(root_noise_epsilon=.25),agent(root_noise_epsilon=.25)
    sampling = random.Random(42)
    symmetry = driver.SymmetryAugmenter(.5)
    before = driver.rng_states(a.rng,sampling,symmetry)
    state = a.root_noise.rng_state()
    priors = (1/7,)*7
    for _ in range(10):
        assert np.array_equal(a.root_noise.mix(priors,range(7))[0],b.root_noise.mix(priors,range(7))[0])
    assert driver.rng_states(a.rng,sampling,symmetry) == before
    assert a.root_noise.domain != symmetry.domain and a.root_noise.seed_digest != symmetry.seed_digest
    replay = RootDirichletNoise(.25)
    replay._rng.bit_generator.state = json.loads(json.dumps(state,allow_nan=False))
    c = RootDirichletNoise(.25)
    for _ in range(10):
        assert np.array_equal(replay.mix(priors,range(7))[0],c.mix(priors,range(7))[0])
    assert replay.rng_state() == a.root_noise.rng_state()
    assert RootDirichletNoise(.25,seed=43).seed_digest != a.root_noise.seed_digest
    json.loads(json.dumps(dict(provenance=a.root_noise.record(),
                              states=driver.rng_states(a.rng,sampling,symmetry,a.root_noise)),allow_nan=False))


@pytest.mark.parametrize('temperature', [0.,1.,2.5])
def test_repeatable_search_and_unchanged_temperature_example_symmetry(temperature):
    a,b = agent(root_noise_epsilon=.25,temperature=temperature),agent(root_noise_epsilon=.25,temperature=temperature)
    game = position([0,1,0,1,0,2])
    for _ in range(3):
        x,y = a.search(game),b.search(game)
        assert (x.move,x.visits,x.policy,x.root_prior,x.root_noise_sample) == (y.move,y.visits,y.policy,y.root_prior,y.root_noise_sample)
        assert np.array_equal(x.policy,visit_policy(x.visits,temperature))
    complete_game = deepcopy(game)
    complete_game.make_move(0)
    example = finalize_examples([capture_example(x)],complete_game)[0]
    mirror = reflect_completed_example(example)
    assert mirror.policy == example.policy[::-1] and mirror.outcome == example.outcome
    assert reflect_completed_example(mirror) == example
    augmenter = driver.SymmetryAugmenter(.5)
    batch,flags = augmenter.batch([example]*32)
    payload = dict(example=asdict(example), batch=[asdict(e) for e in batch], flags=flags,
                   root_noise=dict(epsilon=x.root_noise_epsilon,alpha=x.root_dirichlet_alpha,
                                   sample=x.root_noise_sample,priors=x.root_prior), provenance=a.root_noise.record())
    json.loads(json.dumps(payload,allow_nan=False))


def test_no_noise_diagnostics_or_strength_evaluation(monkeypatch):
    calls=[]
    original = RootDirichletNoise.mix
    def checked(self,priors,moves):
        assert self.epsilon == 0
        state = self.rng_state()
        value = original(self,priors,moves)
        assert self.rng_state() == state and self.draws == 0
        calls.append(1)
        return value
    monkeypatch.setattr(RootDirichletNoise,'mix',checked)
    inference = agent().inference
    nested = snapshot(inference,dict(completed_games=0,updates=0,losses=[]))
    campaign = evaluate(dict(initial=inference,final=inference))
    assert campaign['completed_games'] == 144 and calls
    json.loads(json.dumps(dict(snapshot=nested,evaluation=campaign),allow_nan=False))
