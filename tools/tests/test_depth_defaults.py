"""Exercise the actual launcher-owned tables through its DRY interface."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'VLLM_GLM5_DFLASH_'
SINGLE = PREFIX + 'ADAPTIVE_K_COSTS'
MULTI = SINGLE + '_MULTI'
TABLES = {
    'tp4': ('1.0000,1.1009,1.1631,1.2214,1.2718', '1.0000,1.1132,1.1958,1.2214,1.2718'),
    'pp4': ('1.0000,1.0894,1.1375,1.2001,1.2239', '1.0000,1.0889,1.1194,1.2001,1.2239'),
}


def launch(layout, **overrides):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(PREFIX) and k not in ('PP', 'TP', 'LAYOUT', 'SPEC_N', 'EXTRA_ARGS')}
    env.update(DRY='1', LAYOUT=layout, VLLM_GLM5_MARLIN_DECODE_CUDA='0',
               MODEL='/unused/target', DFLASH_MODEL='/unused/drafter')
    env.update(overrides)
    result = subprocess.run(['bash', str(ROOT / 'serve.sh')], env=env,
                            capture_output=True, text=True, check=True, timeout=30)
    return dict(line.strip().split('=', 1) for line in result.stdout.splitlines()
                if line.strip().startswith((SINGLE + '=', MULTI + '=')))


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('depths,base,lo,hi', [('7,5', '3', 3, 7), ('5,5', '3', 3, 5),
                                            ('5,5', '4', 4, 5), ('7,6', '5', 5, 7)])
def test_defaults_are_indexed_by_absolute_depth(layout, depths, base, lo, hi):
    values = launch(layout, **{PREFIX + 'ADAPTIVE_K_DEPTHS': depths, 'SPEC_N': base})
    for key, table in zip((SINGLE, MULTI), TABLES[layout]):
        assert values[key] == ','.join(table.split(',')[lo - 3:hi - 2])


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('key', [SINGLE, MULTI])
@pytest.mark.parametrize('explicit', ['', '1,2,3', '1,0,2', 'not,a,table'])
def test_explicit_values_remain_owned_by_the_user(layout, key, explicit):
    values = launch(layout, **{PREFIX + 'ADAPTIVE_K_DEPTHS': '5,5', key: explicit})
    assert values[key] == explicit
    other = MULTI if key == SINGLE else SINGLE
    table = TABLES[layout][1 if other == MULTI else 0]
    assert values[other] == ','.join(table.split(',')[:3])
