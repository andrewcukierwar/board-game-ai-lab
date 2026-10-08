"""Public implementation identifiers, not semantic-version compatibility promises.

See docs/evaluation-provenance.md for meanings and bump rules.
"""
import re

API_PROVENANCE_VERSION = 1
API_CONTRACT_VERSION = 2
CONNECT4_ENGINE_VERSION = 1
RANDOM_VERSION = 2
NEGAMAX_VERSION = 2
MCTS_VERSION = 2
STOCHASTIC_SEED_VERSION = 1


def public_provenance(source_commit=None):
    # Only a full explicitly configured Git object ID is public. Never leak an
    # arbitrary environment/config value or infer a commit from server paths.
    commit = source_commit if isinstance(source_commit, str) and re.fullmatch(
        r'[0-9a-f]{40}|[0-9a-f]{64}', source_commit) else None
    return dict(api_provenance_version=API_PROVENANCE_VERSION,
                api_contract_version=API_CONTRACT_VERSION,
                connect4_engine_version=CONNECT4_ENGINE_VERSION,
                agents=dict(random=RANDOM_VERSION, negamax=NEGAMAX_VERSION, mcts=MCTS_VERSION),
                stochastic_seed_version=STOCHASTIC_SEED_VERSION,
                source_commit=commit)
