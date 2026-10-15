"""Deployment safety exercised ONLY with fake Docker/git/curl, never production."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHA = 'a' * 40


def harness(tmp_path, failure='', existing=None):
    root = tmp_path / 'checkout'
    (root / 'scripts').mkdir(parents=True)
    shutil.copy(ROOT / 'scripts/deploy-mac-mini-api.sh', root / 'scripts')
    (root / 'private.env').write_text('SECRET=never-print-me\nALPHAZERO_ENABLED=true\n')
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    state = tmp_path / 'containers.json'
    state.write_text(json.dumps(existing or {'bgai-api': {'running': True, 'image': 'original'}}))
    log = tmp_path / 'calls.jsonl'
    (binaries / 'real-python').symlink_to(sys.executable)
    fake = '''#!/usr/bin/env real-python
import json, os, sys
from pathlib import Path
args=sys.argv[1:]; name=Path(sys.argv[0]).name
p=Path(os.environ['FAKE_STATE']); state=json.loads(p.read_text()); initial=json.dumps(state); failure=os.environ['FAKE_FAILURE']
with open(os.environ['FAKE_LOG'],'a') as f: f.write(json.dumps([name,*args])+'\\n')
# The script pipes fakes concurrently (curl | python3), so read-only calls must
# never rewrite state and real changes land atomically (no truncated reads).
def finish(code=0):
 if json.dumps(state)!=initial:
  tmp=p.with_name(f'{p.name}.{os.getpid()}.tmp'); tmp.write_text(json.dumps(state)); os.replace(tmp,p)
 sys.exit(code)
if name=='git':
 if args[:2]==['status','--porcelain']: print(' M dirty' if failure=='dirty' else '')
 elif args==['rev-parse','HEAD']: print('a'*40)
 finish()
if name=='curl':
 print('OK' if args[-1].endswith('/health') else json.dumps({'source_commit':'a'*40})); finish()
if name=='python3':
 if args[0].endswith('smoke-api-release.py'):
  if failure=='candidate-smoke' and 'http://127.0.0.1:8001' in args: finish(1)
  if failure=='cutover-smoke' and 'http://127.0.0.1:8000' in args: finish(1)
  finish()
 os.execv('REAL_PYTHON', ['REAL_PYTHON', *args])
if args[:2]==['container','inspect']: finish(0 if args[2] in state else 1)
if args[:2]==['image','inspect']: print('amd64' if failure=='arch' else 'arm64'); finish()
if args[:2]==['buildx','build']: finish(1 if failure=='build' else 0)
if args[0] in ('create','run'):
 container=args[args.index('--name')+1]
 if failure=='cutover-run' and container=='bgai-api': finish(1)
 state[container]={'running':args[0]=='run', 'image':args[-1]}; print(container); finish()
if args[0]=='rename':
 if args[2] in state: finish(1)  # like Docker, never clobber an existing name
 state[args[2]]=state.pop(args[1]); finish()
if args[0]=='stop': state[args[1]]['running']=False; finish()
if args[0]=='start':
 if failure=='candidate-start' and args[1]=='bgai-api-candidate': finish(1)
 state[args[1]]['running']=True; finish()
if args[:2]==['rm','-f']: state.pop(args[2],None); finish()
finish()
'''.replace('REAL_PYTHON', sys.executable)
    for name in ['git', 'docker', 'curl', 'python3']:
        path = binaries / name
        path.write_text(fake)
        path.chmod(0o755)
    env = {**os.environ, 'PATH': str(binaries) + ':' + os.environ['PATH'],
           'BGAI_API_ENV_FILE': str(root / 'private.env'), 'FAKE_STATE': str(state),
           'FAKE_LOG': str(log), 'FAKE_FAILURE': failure}
    def run(mode=None):
        result = subprocess.run(['bash', str(root / 'scripts/deploy-mac-mini-api.sh'), *([] if mode is None else [mode])],
                                env=env, text=True, capture_output=True, timeout=10)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        assert 'never-print-me' not in result.stdout + result.stderr
        # A crashing fake must never stand in for the simulated failure under test.
        assert 'Traceback' not in result.stderr, result.stderr
        return result, json.loads(state.read_text()), calls
    return run


# The call each simulated failure fires at; asserting it was reached proves the
# test exercised that stage rather than failing earlier for an unrelated reason.
FAILURE_CALL = {
    'dirty': ['git', 'status', '--porcelain'],
    'build': ['docker', 'buildx', 'build'],
    'arch': ['docker', 'image', 'inspect'],
    'candidate-start': ['docker', 'start', 'bgai-api-candidate'],
    'candidate-smoke': ['python3', 'scripts/smoke-api-release.py', '--base', 'http://127.0.0.1:8001'],
    'cutover-run': ['docker', 'run'],
    'cutover-smoke': ['python3', 'scripts/smoke-api-release.py', '--base', 'http://127.0.0.1:8000'],
}


def reached(calls, prefix):
    return any(c[:len(prefix)] == prefix for c in calls)


@pytest.mark.parametrize('mode', [None, 'lite', 'alphazero'])
def test_deploy_explicit_mode_and_preserved_rollback(tmp_path, mode):
    result, state, calls = harness(tmp_path)(mode)
    assert result.returncode == 0, result.stderr
    selected = mode or 'lite'
    assert state['bgai-api']['image'] == f'bgai-api:{SHA}-{selected}'
    assert state['bgai-api-rollback'] == {'running': False, 'image': 'original'}
    assert 'bgai-api-candidate' not in state
    build = next(c for c in calls if c[:3] == ['docker', 'buildx', 'build'])
    enabled = selected == 'alphazero'
    assert f'ALPHAZERO_RUNTIME={str(enabled).lower()}' in build
    runs = [c for c in calls if c[0] == 'docker' and c[1] in ('create', 'run')]
    assert len(runs) == 2 and all(f'ALPHAZERO_ENABLED={str(enabled).lower()}' in c for c in runs)
    assert runs[0][runs[0].index('-p') + 1] == '127.0.0.1:8001:8000'
    assert calls.index(next(c for c in calls if c[0] == 'python3' and c[1].endswith('smoke-api-release.py'))) < calls.index(['docker', 'rename', 'bgai-api', 'bgai-api-rollback'])


@pytest.mark.parametrize('failure', ['candidate-smoke', 'candidate-start', 'arch', 'build', 'dirty'])
def test_candidate_failure_never_touches_live_container(tmp_path, failure):
    result, state, calls = harness(tmp_path, failure)('alphazero')
    assert result.returncode != 0 and reached(calls, FAILURE_CALL[failure])
    assert state == {'bgai-api': {'running': True, 'image': 'original'}}
    assert not any(c[:2] in (['docker', 'stop'], ['docker', 'rename']) for c in calls)


@pytest.mark.parametrize('failure', ['cutover-smoke', 'cutover-run'])
def test_failed_cutover_restores_original_container_environment_and_image(tmp_path, failure):
    result, state, calls = harness(tmp_path, failure)('alphazero')
    assert result.returncode != 0 and reached(calls, FAILURE_CALL[failure])
    assert state == {'bgai-api': {'running': True, 'image': 'original'}}
    assert ['docker', 'start', 'bgai-api'] in calls


@pytest.mark.parametrize('name', ['bgai-api-candidate', 'bgai-api-rollback'])
def test_refuses_foreign_candidate_or_existing_rollback(tmp_path, name):
    initial = {'bgai-api': {'running': True, 'image': 'original'}, name: {'running': False, 'image': 'foreign'}}
    result, state, calls = harness(tmp_path, existing=initial)('alphazero')
    assert result.returncode != 0 and state == initial
    assert ['docker', 'container', 'inspect', name] in calls
    assert not any(c[:2] in (['docker', 'run'], ['docker', 'rm']) for c in calls)


def test_unknown_mode_fails_before_docker(tmp_path):
    result, _, calls = harness(tmp_path)('enable-everything')
    assert result.returncode != 0 and not calls
