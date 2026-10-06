from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import env_effective as effective

ENV_SOURCE = '''environment_variables = {
    "VLLM_GLM5_DECODE_KDA_STEP_TILE": lambda: bool(int(os.getenv("VLLM_GLM5_DECODE_KDA_STEP_TILE", "0"))),
    "VLLM_GLM5_DFLASH_BOUNDARY_CACHE": lambda: bool(int(os.getenv("VLLM_GLM5_DFLASH_BOUNDARY_CACHE", "0"))),
    "VLLM_GLM5_DFLASH_CONFIDENCE_LOG": lambda: os.getenv("VLLM_GLM5_DFLASH_CONFIDENCE_LOG", ""),
}'''
PARALLEL_SOURCE = '''class ParallelConfig:
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
'''
SCHEDULER_SOURCE = '''class SchedulerConfig:
    long_prefill_token_threshold: int = Field(default=0, ge=0)
'''


@pytest.fixture
def leg(tmp_path, monkeypatch):
    monkeypatch.setattr(effective, "git_source", lambda fork, name: {
        "vllm/envs.py": ENV_SOURCE,
        "vllm/config/parallel.py": PARALLEL_SOURCE,
        "vllm/config/scheduler.py": SCHEDULER_SOURCE,
    }[name])
    actual, command, _ = effective.dry_run(ROOT, "pp4")
    # The full fixture contains every captured key; no environment inheritance.
    actual.pop("VLLM_GLM5_DECODE_KDA_STEP_TILE")
    actual["VLLM_GLM5_DFLASH_CONFIDENCE_LOG"] = ""
    for key in ("tensor_parallel_size", "long_prefill_token_threshold"):
        command.pop(key)
    return tmp_path / "effective_env_private.json", actual, command


def run(leg):
    path, environment, command = leg
    path.write_text(json.dumps(environment))
    path.with_name("serve.log").write_text("non-default args: " + repr(command) + "\n")
    return effective.check_effective(ROOT, path, "pp4", path.parent)


def test_complete_pp4_environment_uses_proven_zero_default(leg, capsys):
    assert run(leg) == 0
    output = capsys.readouterr().out
    assert "step_tile=0" in output
    assert "VLLM_GLM5_DECODE_KDA_STEP_TILE (explicit '0' equals fork default False" in output
    assert "Full environment match: PASS (pp4; 0 mismatches)" in output


@pytest.mark.parametrize("key,value", [
    ("UNEXPECTED_RUNTIME_SETTING", "1"),
    ("UNEXPECTED_RUNTIME_SETTING", "/some/path"),
    ("VLLM_GLM5_DECODE_KDA_STEP_TILE", "1"),
    ("VLLM_GLM5_DFLASH_CONFIDENCE_LOG", "diagnostic"),
    ("VLLM_GLM5_DFLASH_BOUNDARY_CACHE", "0"),
    ("EXTRA_ARGS", "--enforce-eager"),
    ("NCCL_P2P_DISABLE", "1"),
])
def test_unlisted_validation_keys_and_changed_values_fail(leg, capsys, key, value):
    leg[1][key] = value
    assert run(leg) == 1
    assert f"environment: {key}: validation=" in capsys.readouterr().out


def test_extra_recipe_setting_cannot_hide_behind_missing_reference_key(leg, capsys):
    leg[1].pop("VLLM_GLM5_DFLASH_BOUNDARY_CACHE")
    assert run(leg) == 1
    assert "environment: VLLM_GLM5_DFLASH_BOUNDARY_CACHE: validation=None" in capsys.readouterr().out


def test_changed_pinned_default_invalidates_equivalence(leg, monkeypatch):
    original = effective.git_source
    monkeypatch.setattr(effective, "git_source", lambda fork, name:
                        original(fork, name).replace('STEP_TILE", "0"', 'STEP_TILE", "1"'))
    assert run(leg) == 1


@pytest.mark.parametrize("key,value", [
    ("max_num_batched_tokens", 2048),
    ("pipeline_parallel_size", 1),
    ("tensor_parallel_size", 4),
    ("long_prefill_token_threshold", 512),
    ("speculative_config", {"method": "dflash", "num_speculative_tokens": 7}),
    ("enforce_eager", True),
])
def test_command_changes_and_unexpected_flags_fail(leg, capsys, key, value):
    leg[2][key] = value
    assert run(leg) == 1
    assert f"command: {key}: validation=" in capsys.readouterr().out


def test_only_documented_paths_and_command_exclusions_pass(leg):
    leg[1]["MODEL"] = "/other/model"
    leg[1]["EXTRA_ARGS"] = "--enable-prompt-tokens-details"
    leg[2].update(model="/other/model", model_tag="/other/model", host="0.0.0.0",
                  port=9000, enable_prompt_tokens_details=True)
    leg[2]["speculative_config"]["model"] = "/other/drafter"
    assert run(leg) == 0


def test_empty_or_inconsistent_serve_log_fails(leg):
    path, environment, _ = leg
    path.write_text(json.dumps(environment))
    for text in ("", "non-default args: {}\nnon-default args: {'enforce_eager': True}\n"):
        path.with_name("serve.log").write_text(text)
        with pytest.raises(ValueError, match="consistent non-default args"):
            effective.check_effective(ROOT, path, "pp4", path.parent)


@pytest.mark.parametrize("layout,default,override", [("tp4", "1", "0"), ("pp4", "0", "1")])
def test_step_tile_defaults_banners_and_explicit_overrides(layout, default, override, tmp_path):
    _, _, stdout = effective.dry_run(ROOT, layout)
    assert f"step_tile={default}" in stdout
    # Exercise the launcher's explicit override using the same CPU fixture.
    import os
    import subprocess
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "_ampere_marlin_C.py").touch()
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python").symlink_to(sys.executable)
    result = subprocess.run(["bash", str(ROOT / "serve.sh")], env={
        "PATH": os.defpath, "DRY": "1", "P2P": "force", "LAYOUT": layout,
        "VENV": str(tmp_path), "PYTHONPATH": str(tmp_path),
        "VLLM_GLM5_DECODE_KDA_STEP_TILE": override,
    }, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"step_tile={override}" in result.stdout
    assert f"  VLLM_GLM5_DECODE_KDA_STEP_TILE={override}" in result.stdout


@pytest.mark.parametrize("key,value,changed", [
    ("FAIR_CHUNK", "384", "385"),
    ("FAIR_PARTIAL", "2", "3"),
    ("FAIR_PREFILL", "1", "0"),
    ("PREFILL_CAP", "0", "1"),
])
def test_launcher_aliases_are_resolved_and_changed_values_fail(leg, key, value, changed):
    leg[1][key] = value
    assert run(leg) == 0
    leg[1][key] = changed
    assert run(leg) == 1


@pytest.mark.parametrize("field,value", [
    ("prefill_chunk_with_decodes", 385),
    ("max_num_partial_prefills", 3),
    ("long_prefill_token_threshold", 1),
])
def test_matching_alias_does_not_hide_changed_live_command(leg, field, value):
    leg[1].update(FAIR_CHUNK="384", FAIR_PARTIAL="2", FAIR_PREFILL="1", PREFILL_CAP="0")
    leg[2][field] = value
    assert run(leg) == 1


def test_only_harness_pci_bus_id_is_allowed(leg):
    leg[1]["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    assert run(leg) == 0
    leg[1]["CUDA_DEVICE_ORDER"] = "FASTEST_FIRST"
    assert run(leg) == 1


def test_recipe_cuda_device_order_is_not_allowlisted(leg, monkeypatch):
    original = effective.dry_run
    def with_order(*args):
        actual, command, stdout = original(*args)
        actual["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        return actual, command, stdout
    monkeypatch.setattr(effective, "dry_run", with_order)
    leg[1]["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    assert run(leg) == 1


def test_tp4_selector_is_inert_only_with_pp4_whole_expert_path(leg):
    leg[1]["VLLM_GLM5_TP4_MARLIN_PREFILL"] = "1"
    assert run(leg) == 0
    leg[1]["VLLM_GLM5_PP_MARLIN_PREFILL"] = "0"
    assert run(leg) == 1


def test_tp4_selector_mismatch_still_fails_tp4():
    actual = {"VLLM_GLM5_TP4_MARLIN_PREFILL": "0", "VLLM_GLM5_PP_MARLIN_PREFILL": "1"}
    expected = {"VLLM_GLM5_TP4_MARLIN_PREFILL": "1", "VLLM_GLM5_PP_MARLIN_PREFILL": "1"}
    assert effective.compare_environment(actual, expected, {}, "", "tp4")
