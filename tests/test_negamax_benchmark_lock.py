"""Lock lifecycle tests with only the lock/cache path redirected into tmp_path.

These are sleeping correctness subprocesses, not performance workloads. The
production wrapper always uses the mandated shared path without any override.
"""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def isolated_wrapper(tmp_path):
    source=Path('scripts/with_benchmark_lock.sh').read_text()
    source=source.replace('lock_dir="$HOME/.cache/bgai-laptop-benchmark.lock"',
                          f'lock_dir="{tmp_path}/lock"')
    source=source.replace('mkdir -p "$HOME/.cache"',f'mkdir -p "{tmp_path}/cache"')
    script=tmp_path/'wrapper.sh'
    script.write_text(source)
    return script,tmp_path/'lock'


@pytest.mark.parametrize('exit_code',[0,7])
def test_success_and_failure_release_owned_lock(tmp_path,exit_code):
    script,lock=isolated_wrapper(tmp_path)
    command=[sys.executable,'-c',f'import sys; sys.exit({exit_code})']
    result=subprocess.run(['sh',str(script),*command],capture_output=True,text=True)
    assert result.returncode==exit_code
    assert not lock.exists()


def test_busy_lock_is_not_stolen_and_child_not_started(tmp_path):
    script,lock=isolated_wrapper(tmp_path)
    lock.mkdir()
    owner=f'pid={os.getpid()}\nbranch=research/mcts-v3\nstart=test\n'
    (lock/'owner').write_text(owner)
    marker=tmp_path/'started'
    child=f'from pathlib import Path; Path({str(marker)!r}).write_text("started")'
    result=subprocess.run(['sh',str(script),sys.executable,'-c',child],capture_output=True,text=True)
    assert result.returncode==75
    assert (lock/'owner').read_text()==owner
    assert not marker.exists()


def test_signal_does_not_release_before_child_finishes(tmp_path):
    script,lock=isolated_wrapper(tmp_path)
    ready,finish=tmp_path/'ready',tmp_path/'finish'
    child=('import time; from pathlib import Path; '
           f'Path({str(ready)!r}).write_text("ready"); '
           f'finish=Path({str(finish)!r}); '
           '\nwhile not finish.exists(): time.sleep(.01)')
    process=subprocess.Popen(['sh',str(script),sys.executable,'-c',child],
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline:
            time.sleep(.01)
        assert ready.exists()
        owner=(lock/'owner').read_text()
        assert f'pid={process.pid}\n' in owner
        process.send_signal(signal.SIGTERM)
        time.sleep(.05)
        assert process.poll() is None
        assert (lock/'owner').read_text()==owner
        finish.write_text('finish')
        stdout,stderr=process.communicate(timeout=5)
        assert process.returncode==0,(stdout,stderr)
        assert not lock.exists()
    finally:
        finish.write_text('finish')
        if process.poll() is None:
            process.communicate(timeout=5)
