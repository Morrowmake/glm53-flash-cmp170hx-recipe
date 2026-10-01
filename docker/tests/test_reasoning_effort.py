import os
from pathlib import Path
import shlex
import subprocess
import json

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize('effort', ['', 'low', 'high', 'max'])
def test_default_effort_dry_command(tmp_path, effort):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', DRY='1', VENV=str(tmp_path/'missing'),
               MODEL='/target', DFLASH_MODEL='/draft', DEFAULT_REASONING_EFFORT=effort,
               VLLM_GLM5_MARLIN_DECODE_CUDA='0', VLLM_GLM5_MARLIN_PREFILL_CUDA='0', EXTRA_ARGS='')
    run = subprocess.run(['bash', str(ROOT/'serve.sh')], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    args = shlex.split(run.stdout.split('serve.sh: DRY=1, command:\n')[1])
    flag = '--default-chat-template-kwargs'
    if effort:
        assert args.count(flag) == 1
        assert json.loads(args[args.index(flag)+1]) == dict(reasoning_effort=effort)
        assert f'[reasoning-effort] default={effort}' in run.stdout
    else:
        assert flag not in args
        assert '[reasoning-effort] template default' in run.stdout


def test_invalid_effort_and_explicit_cli_precedence(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', DRY='1', VENV=str(tmp_path/'missing'),
               MODEL='/target', DFLASH_MODEL='/draft', DEFAULT_REASONING_EFFORT='invalid',
               VLLM_GLM5_MARLIN_DECODE_CUDA='0', VLLM_GLM5_MARLIN_PREFILL_CUDA='0', EXTRA_ARGS='')
    run = subprocess.run(['bash', str(ROOT/'serve.sh')], env=env, capture_output=True, text=True)
    assert run.returncode == 2 and 'must be empty, low, high or max' in run.stderr
    env.update(DEFAULT_REASONING_EFFORT='low', EXTRA_ARGS='--default-chat-template-kwargs={"reasoning_effort":"high"}')
    run = subprocess.run(['bash', str(ROOT/'serve.sh')], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    args = shlex.split(run.stdout.split('serve.sh: DRY=1, command:\n')[1])
    assert sum(arg.startswith('--default-chat-template-kwargs') for arg in args) == 1
    assert '--default-chat-template-kwargs={"reasoning_effort":"high"}' in args
    assert '[reasoning-effort] explicit EXTRA_ARGS defaults' in run.stdout


@pytest.mark.parametrize('effort', ['', 'low'])
def test_container_environment_forwarding(tmp_path, effort):
    (tmp_path/'.env.example').write_text('')
    source = (ROOT/'start.sh').read_text().rsplit('main "$@"', 1)[0]
    source += '\ncontainer_env\n'
    script = tmp_path/'start.sh'
    script.write_text(source)
    run = subprocess.run(['bash', str(script)], capture_output=True, text=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES='', DEFAULT_REASONING_EFFORT=effort))
    assert run.returncode == 0, run.stderr
    assert 'DEFAULT_REASONING_EFFORT=' + effort in run.stdout.splitlines()
