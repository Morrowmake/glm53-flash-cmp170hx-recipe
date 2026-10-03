from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def reference(tmp_path):
    values = {
        "tp4": {
            "MAX_BATCHED": "3463",
            "VLLM_GLM5_MARLIN_DECODE_CUDA": "1",
            "VLLM_GLM5_DECODE_KDA_V2_DEEP": "1",
            "VLLM_GLM5_KDA_RECOVER": "1",
            "VLLM_CUSTOM_ALLREDUCE_FLAGS": "1",
            "VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS":
                "1.0000,1.1009,1.1631,1.2214,1.2718",
        },
        "pp4": {
            "VLLM_PP_LAYER_PARTITION": "13,11,11,10",
            "VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS": "9-64",
            "VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED": "1",
            "GLM5_PP4_DRAFT_WIDTH": "0",
            "VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS_MULTI":
                "1.0000,1.0889,1.1194,1.2001,1.2239",
        },
    }
    for layout in values.values():
        layout.update({
            "VLLM_GLM5_DFLASH_BOUNDARY_CACHE": "1",
            "VLLM_GLM5_DFLASH_SKIP": "0",
            "VLLM_GLM5_DFLASH_DEPTH2": "0",
            "VLLM_GLM5_STATE_INDEX_CHECK": "0",
        })
    return tmp_path / "reference.json", values


def run(reference):
    path, values = reference
    path.write_text(json.dumps(values))
    environment = dict(os.environ, VLLM_GLM5_DFLASH_BOUNDARY_CACHE="0",
                       VLLM_GLM5_DECODE_KDA_V2_DEEP="0")
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/check-env-match.py"), str(path)],
        env=environment, text=True, capture_output=True,
    )


def test_defaults_match_without_inheriting_shell_overrides(reference):
    result = run(reference)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Environment match: PASS" in result.stdout


@pytest.mark.parametrize("layout,key,value", [
    ("tp4", "VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS", "1,2,3,4,5"),
    ("pp4", "VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS", "9-32"),
    ("pp4", "VLLM_GLM5_STATE_INDEX_CHECK", "1"),
])
def test_reports_changed_validation_value(reference, layout, key, value):
    reference[1][layout][key] = value
    result = run(reference)
    assert result.returncode == 1
    assert f"{layout}: {key}: expected" in result.stdout


def test_lists_diagnostic_and_path_exclusions(reference):
    reference[1]["tp4"].update({
        "VLLM_GLM5_DFLASH_CONFIDENCE_LOG": "diagnostic",
        "VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR": "/validation/cache",
    })
    result = run(reference)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ignored VLLM_GLM5_DFLASH_CONFIDENCE_LOG (diagnostic only)" in result.stdout
    assert "ignored VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR (path valued)" in result.stdout
