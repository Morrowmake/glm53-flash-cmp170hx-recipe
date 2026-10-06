"""Exercise the actual launcher-owned tables through its DRY interface."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(os.environ.get('RECIPE_TEST_ROOT', Path(__file__).resolve().parents[2]))
PREFIX = 'VLLM_GLM5_DFLASH_'
SINGLE = PREFIX + 'ADAPTIVE_K_COSTS'
MULTI = SINGLE + '_MULTI'
TABLES = {
    'tp4': ('1.0000,1.1009,1.1631,1.2214,1.2718', '1.0000,1.1132,1.1958,1.2214,1.2718'),
    'pp4': ('1.0000,1.0894,1.1375,1.2001,1.2239', '1.0000,1.0889,1.1194,1.2001,1.2239'),
}
K = PREFIX + 'ADAPTIVE_K'
ACCEPT = K + '_ACCEPT'
WIDTH = PREFIX + 'ADAPTIVE_DRAFT_WIDTH'
DEPTHS = K + '_DEPTHS'
DEPTH_CASES = [(base, one, two) for base, one, two in itertools.product(range(3, 8), repeat=3)
               if max(one, two) > base]


@pytest.fixture(autouse=True)
def cpu_launcher(tmp_path, monkeypatch):
    package = tmp_path / 'vllm'
    package.mkdir()
    (package / '__init__.py').touch()
    (package / '_ampere_marlin_C.py').touch()
    (tmp_path / 'bin').mkdir()
    (tmp_path / 'bin/python').symlink_to(sys.executable)
    monkeypatch.setenv('VENV', str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    monkeypatch.setenv('RECIPE_CACHE_ROOT', str(tmp_path / 'cache'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('P2P', 'force')


def launch(layout, root=ROOT, **overrides):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(PREFIX) and k not in
           ('PP', 'TP', 'LAYOUT', 'SPEC_N', 'EXTRA_ARGS', 'GLM5_PP4_DRAFT_WIDTH')}
    env.update(DRY='1', LAYOUT=layout, VLLM_GLM5_MARLIN_DECODE_CUDA='0',
               MODEL='/unused/target', DFLASH_MODEL='/unused/drafter')
    env.update(overrides)
    result = subprocess.run(['bash', str(root / 'serve.sh')], env=env,
                            capture_output=True, text=True, check=True, timeout=30)
    dumped = dict(line.strip().split('=', 1) for line in result.stdout.splitlines()
                  if line.strip().startswith(('VLLM_', 'GLM5_', 'NCCL_', 'PYTORCH_',
                                              'PYTHONPATH=', 'FLASHINFER_', 'CUDA_HOME=')))
    command = next(line for line in result.stdout.splitlines()
                   if ' --served-model-name ' in line)
    args = shlex.split(command)
    raw, pp = None, None
    for index, arg in enumerate(args):
        if arg == '--speculative-config':
            raw = json.loads(args[index + 1])
        elif arg.startswith('--speculative-config='):
            raw = json.loads(arg.split('=', 1)[1])
        elif arg == '--pipeline-parallel-size':
            pp = int(args[index + 1])
        elif arg.startswith('--pipeline-parallel-size='):
            pp = int(arg.split('=', 1)[1])
    return dumped, raw, pp, result.stdout


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('depths,base,lo,hi', [('7,5', '3', 3, 7), ('5,5', '3', 3, 5),
                                            ('5,5', '4', 4, 5), ('7,6', '5', 5, 7)])
def test_defaults_are_indexed_by_absolute_depth(layout, depths, base, lo, hi):
    values, _, _, _ = launch(layout, **{DEPTHS: depths, 'SPEC_N': base})
    for key, table in zip((SINGLE, MULTI), TABLES[layout]):
        assert values[key] == ','.join(table.split(',')[lo - 3:hi - 2])


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('key', [SINGLE, MULTI])
@pytest.mark.parametrize('explicit', ['', '1,2,3', '1,0,2', 'not,a,table'])
def test_explicit_values_remain_owned_by_the_user(layout, key, explicit):
    values, _, _, _ = launch(layout, **{DEPTHS: '5,5', key: explicit})
    assert values[key] == explicit
    other = MULTI if key == SINGLE else SINGLE
    table = TABLES[layout][1 if other == MULTI else 0]
    assert values[other] == ','.join(table.split(',')[:3])


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('base,one,two', DEPTH_CASES)
def test_documented_depth_range_dry_run(layout, base, one, two):
    values, raw, _, stdout = launch(layout, **{DEPTHS: f'{one},{two}', 'SPEC_N': str(base)})
    lo, hi = min(base, one, two), max(base, one, two)
    assert raw['num_speculative_tokens'] == base
    for key, table in zip((SINGLE, MULTI), TABLES[layout]):
        assert values[key] == ','.join(table.split(',')[lo - 3:hi - 2])
        assert len(values[key].split(',')) == hi - lo + 1
    assert 'set but off' not in stdout


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('enabled', ['0', '1'])
def test_adaptive_switch_has_coherent_unset_dependents(layout, enabled):
    values, raw, _, _ = launch(layout, **{K: enabled})
    assert values[ACCEPT] == enabled
    assert values[WIDTH] == ('1' if enabled == '1' and layout == 'tp4' else '0')
    assert (values[SINGLE], values[MULTI]) == TABLES[layout]
    assert raw['num_speculative_tokens'] == 3


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('key', [ACCEPT, WIDTH])
@pytest.mark.parametrize('value', ['0', '1', ''])
def test_explicit_dependent_values_are_not_rewritten(layout, key, value):
    values, _, _, _ = launch(layout, **{K: '0', key: value})
    assert values[key] == value


@pytest.mark.parametrize('layout', TABLES)
def test_ordinary_defaults_unchanged(layout):
    current = launch(layout)
    values, raw, _, _ = current
    assert (values[K], values[ACCEPT], values[WIDTH], values[DEPTHS]) == (
        '1', '1', '1' if layout == 'tp4' else '0', '7,5')
    assert (values[SINGLE], values[MULTI]) == TABLES[layout]
    assert raw['num_speculative_tokens'] == 3
    reference = os.environ.get('RECIPE_REFERENCE_ROOT')
    if reference:
        previous = launch(layout, root=Path(reference))
        assert current[:3] == previous[:3]


@pytest.fixture(scope='module')
def engine():
    """Native-runtime checks use the installed fork, without loading a model."""
    source = os.environ.get('RECIPE_TEST_ENGINE')
    if source:
        sys.path.insert(0, source)
    module = pytest.importorskip('vllm.config.speculative',
                                reason='install the pinned native fork for CPU engine-contract tests')
    import torch
    import vllm.envs as envs
    from vllm.v1.spec_decode.dynamic.adaptive_k import AdaptiveKConfig
    assert not torch.cuda.is_initialized()
    return module, envs, AdaptiveKConfig, torch


def consume(engine, result):
    module, envs, policy_type, torch = engine
    dumped, raw, pp, _ = result
    environment = {k: v for k, v in os.environ.items() if not k.startswith(PREFIX)}
    environment.update(dumped)
    config = object.__new__(module.SpeculativeConfig)
    config.method = raw['method']
    config.num_speculative_tokens = raw['num_speculative_tokens']
    config.adaptive_k = raw.get('adaptive_k')
    config.num_speculative_tokens_per_batch_size = raw.get('num_speculative_tokens_per_batch_size')
    config.enable_adaptive_verification = raw.get('enable_adaptive_verification', False)
    config.target_parallel_config = SimpleNamespace(pipeline_parallel_size=pp)
    before = dict(config.__dict__)
    with patch.dict(os.environ, environment, clear=True), patch.object(
            module.logger, 'warning_once') as warnings:
        envs.disable_envs_cache()
        config._maybe_enable_glm5_load_adaptive_depth()
        policy = (policy_type.from_dict(config.adaptive_k, config.num_speculative_tokens)
                  if config.adaptive_k is not None else None)
    assert not torch.cuda.is_initialized()
    return config, policy, before, warnings.call_args_list


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('base,one,two', DEPTH_CASES)
def test_documented_depth_range_in_engine(engine, layout, base, one, two):
    result = launch(layout, **{DEPTHS: f'{one},{two}', 'SPEC_N': str(base)})
    _, policy, _, warnings = consume(engine, result)
    assert not warnings, warnings
    assert policy.accept
    assert policy.allowed == tuple(range(min(base, one, two), max(base, one, two) + 1))
    assert policy.accept_costs == tuple(map(float, result[0][SINGLE].split(',')))
    assert policy.accept_costs_multi == tuple(map(float, result[0][MULTI].split(',')))


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('accept,width', list(itertools.product(['0', '1'], repeat=2)))
def test_documented_accept_and_width_switches(engine, layout, accept, width):
    result = launch(layout, **{DEPTHS: '5,5', ACCEPT: accept, WIDTH: width})
    config, policy, _, warnings = consume(engine, result)
    assert not warnings, warnings
    assert policy.accept == (accept == '1')
    assert config.adaptive_k.get('draft_by_load', False) == (width == '1')


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('base', range(3, 8))
def test_fixed_depth_off_path_is_inert(engine, layout, base):
    config, policy, before, warnings = consume(
        engine, launch(layout, **{K: '0', 'SPEC_N': str(base)}))
    assert not warnings, warnings
    assert policy is None
    assert config.__dict__ == before
    explicit, explicit_policy, explicit_before, explicit_warnings = consume(
        engine, launch(layout, **{K: '0', 'SPEC_N': str(base), ACCEPT: '1', WIDTH: '1'}))
    assert explicit_warnings
    assert explicit_policy is None
    assert explicit.__dict__ == explicit_before
