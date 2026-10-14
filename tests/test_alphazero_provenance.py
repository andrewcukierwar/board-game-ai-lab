"""Phase 4D.4K-A torch-free provenance locks: ported research sources, artifact and disabled path."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'models/alphazero/az2-hc-g6-cnn-s42-v1.json'
ARTIFACT = ROOT / 'models/alphazero/az2-hc-g6-cnn-s42-v1.pt'
ARTIFACT_SHA256 = '4c3cf26aa58495f43094436f231486969658d70b94b246a912a5bb54756dc528'
SOURCE_CHECKPOINT_SHA256 = 'ba2bde2ec71b3ac7486046d1cc26ab591bb59d6b764193ece08c6d6a6f24f58b'
# Byte-identical copies of research commit 99ceb370cf9c197627f27e86099976304597902e.
RESEARCH_SOURCES = {
    'games/connect4/alphazero_v2/tactical_guard.py': 'd2ee54ba1fe9ddbcc4641538e2f5dd89a2435234ff194d6909bed72096632d04',
    'games/connect4/alphazero_v2/bitboard.py': '61a262e45191b1e22b57a224f848668315b60ddd5d7785d620c3dd4de2965229',
    'games/connect4/alphazero_v2/network.py': '097712ae8a304a3d0ca38f4ef71fb6aeaf0a7ff8808e532ffd5475473f80ee5c',
    'games/connect4/alphazero_v2/search.py': '49d719dfb30caf97ab562502584b4cd0ca3b3a8a9f82fd9c986a91f0c6ba993d',
    'games/connect4/alphazero_v2/data.py': 'dbe4fec9b4d3300943a175a52d07a8f9cdbbde159f4d102a69c4e93d26f7e254',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize('path,digest', sorted(RESEARCH_SOURCES.items()))
def test_research_sources_are_byte_identical(path, digest):
    assert sha256(ROOT / path) == digest


def test_manifest_binds_artifact_to_research_checkpoint():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['artifact'] == {'file': ARTIFACT.name, 'sha256': ARTIFACT_SHA256,
                                    'size_bytes': ARTIFACT.stat().st_size}
    assert sha256(ARTIFACT) == ARTIFACT_SHA256 and ARTIFACT.stat().st_size == 1309171
    source = manifest['source']
    assert (source['seed'], source['generation'], source['phase'], source['checkpoint_sha256']) == (
        42, 6, '4D.4H-C', SOURCE_CHECKPOINT_SHA256)
    assert source['research_commit'] == '99ceb370cf9c197627f27e86099976304597902e'
    assert manifest['tactical_guard']['source_sha256'] == RESEARCH_SOURCES['games/connect4/alphazero_v2/tactical_guard.py']
    assert manifest['search']['simulations'] == 256 and manifest['search']['exploration'] == 1.41
    assert manifest['search']['root_noise'] is None and manifest['search']['temperature'] == 0.0
    assert all(v == 'PASS' or v.startswith('PASS') for v in manifest['export']['gates'].values())


def test_disabled_backend_never_imports_torch():
    code = ('import sys; from api.app import create_app; app = create_app({"TESTING": True}); '
            'c = app.test_client(); '
            'r = c.post("/v1/connect4/start_game", json={"player2": {"type": "alphazero"}}); '
            'assert r.status_code == 400 and r.json["code"] == "invalid_agent", r.json; '
            'assert c.get("/v1/connect4/agents/alphazero").json["code"] == "alphazero_disabled"; '
            'assert "torch" not in sys.modules and "games.connect4.alphazero_v2.production" not in sys.modules')
    env = {'PATH': '/usr/bin:/bin', 'ALPHAZERO_ENABLED': 'false', 'PYTHONDONTWRITEBYTECODE': '1'}
    done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
