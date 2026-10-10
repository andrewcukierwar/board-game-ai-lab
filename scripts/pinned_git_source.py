"""Phase-baseline bytes that survive shallow checkouts and squash merges.

The Negamax harnesses execute and compare sources from immutable phase commits.
Those commits are absent from depth-1 CI checkouts and may become unreachable
after a squash merge. Git's object database remains the first source; otherwise
the byte-identical committed copy is read. Either way the result must match the
SHA-256 pinned here, which equals the digest each phase manifest recorded, so a
missing object can never silently substitute different code or fixtures.
"""
import hashlib
from pathlib import Path
import subprocess

AGENT_PATH = 'games/connect4/agents/negamax_agent.py'
EVIDENCE = 'docs/search-negamax-v2/'
# Baseline agents no longer exist at AGENT_PATH; evidence files are unchanged
# since their phase commit and are their own committed copy.
AGENT_COPIES = {
    '93ee943d8744651064dcf9a3a79108d6ca73af53': EVIDENCE + 'incremental-evaluation/baseline-source.py',
    'f2f57b46c7817bb7324f4390da0bb345ae2fc7e4': EVIDENCE + 'transposition-table/baseline-source.py',
    '2e79b15ecdb1967345a1e66593201f9803c89757': EVIDENCE + 'iterative-deepening/direct-source.py',
}
PINNED = {
    '93ee943d8744651064dcf9a3a79108d6ca73af53': {
        AGENT_PATH: 'a07a0d19bf523c6afc1870c1b2deaab811f210f61aa3198db780a1752e436dc2',
        EVIDENCE + 'deeper-search/candidate.json':
            'da944a03db84412ad39b89127576993d7c4fbd42d7e71a50e3338c9568603feb',
        EVIDENCE + 'deeper-search/profiling-supplement/candidate.json':
            'ccac8356fc57cc7507547908141dbc117091b8c4e0b30911eb5c58b7862e755d',
        EVIDENCE + 'deeper-search/profiling.json':
            '2f5b1d8702d52ab42a137e2969a91ace53037ef4b6cf4a2653705045d59f6704',
        EVIDENCE + 'deeper-search/profiling-supplement/profiling.json':
            'f0a561d495216e792706f7053837ab0199b3413bf8ffbae2476caeab9021aa48',
        EVIDENCE + 'deeper-search/tail-diagnostic.json':
            '964257c3a82dd569009fb01d4aff3e8b28932c13bdd6f070047e2b5201ea57c3',
    },
    'f2f57b46c7817bb7324f4390da0bb345ae2fc7e4': {
        AGENT_PATH: 'ac7618cfe9854e355640f17cbe1dd96d0fcc119c5fdeb30667d3d51cfc4ffa09',
        EVIDENCE + 'incremental-evaluation/manifest.json':
            'aae890614aaed2500519039be57a215eb251f1e72e09548a724facf0a847519b',
        EVIDENCE + 'incremental-evaluation/results.jsonl':
            '8488c279790f84886b1e704fe483454f6af2e1f07f0668706ed52a5ea1c50af9',
    },
    '2e79b15ecdb1967345a1e66593201f9803c89757': {
        AGENT_PATH: 'f46461d73b240c645248d57884fee0b9917c1be289758efe381cf6458a9cb392',
        EVIDENCE + 'transposition-table/manifest.json':
            '2034ed9b1e3b88721353d7d87e4a092ae3555f18f572429f640d0620240354d5',
        EVIDENCE + 'transposition-table/A-results.jsonl':
            'a3c8c80c8f36753956739272ae4d4f318f1da0665f7dab15b857134330c9c8e7',
    },
}


def committed_copy(commit, path):
    return AGENT_COPIES[commit] if path == AGENT_PATH else path


def pinned_source(commit, path):
    """Return ``path`` as of ``commit``; unpinned or mismatching bytes fail closed."""
    if path not in PINNED.get(commit, ()):
        raise ValueError(f'No pinned digest for {commit}:{path}')
    try:
        data = subprocess.check_output(['git', 'show', f'{commit}:{path}'],
                                       stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        data = Path(committed_copy(commit, path)).read_bytes()
    if hashlib.sha256(data).hexdigest() != PINNED[commit][path]:
        raise ValueError(f'{commit}:{path} does not match its pinned SHA-256')
    return data
