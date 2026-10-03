"""CPU-only comparison of complete validation environments and serve arguments."""
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

PIN = "caaf6afe8ee8c29b32ef3b77284656ca87a0937e"
LAB_KEYS = {
    "GPU_LOCK_TOKENS": "validation GPU lock ownership",
    "LANG": "validation shell locale",
    "LOGNAME": "validation login identity",
    "SHLVL": "validation shell nesting",
    "USER": "validation login identity",
    "VLLM_SERVER_DEV_MODE": "validation server diagnostics",
    "GLM5_NCCL_P2P_SYS": "launcher input; NCCL_P2P_LEVEL is compared",
    "LAYOUT": "launcher input; parallel sizes are compared in the command",
    "MAX_LEN": "launcher input; max_model_len is compared in the command",
    "MAX_SEQS": "launcher input; max_num_seqs is compared in the command",
    "GPU_UTIL": "launcher input; gpu_memory_utilization is compared in the command",
    "MM_CAP": "launcher input; multimodal arguments are compared in the command",
}
PATH_KEYS = {
    "CUDA_HOME", "DFLASH_MODEL", "FLASHINFER_WORKSPACE_BASE", "HOME",
    "MODEL", "PATH", "PWD", "PYTHONPATH", "VENV",
}
IGNORED_ARGS = {
    "model", "model_tag", "host", "port", "enable_prompt_tokens_details",
}


def git_source(fork, name):
    return subprocess.check_output(
        ["git", "-C", str(fork), "show", f"{PIN}:{name}"], text=True)


def read_getters(source):
    getters = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(value, ast.Lambda):
                    getters[key.value] = value
    return getters


def evaluate(node, key, values):
    """Interpret only literal getenv defaults and their simple conversions."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Call) and not node.keywords:
        func = node.func
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == "os" and func.attr == "getenv"):
            args = [evaluate(arg, key, values) for arg in node.args]
            if args[0] != key:
                raise ValueError("default depends on another environment key")
            return values.get(key, args[1] if len(args) == 2 else None)
        if isinstance(func, ast.Name) and func.id in {"bool", "int", "float", "str"}:
            cast = {"bool": bool, "int": int, "float": float, "str": str}[func.id]
            return cast(evaluate(node.args[0], key, values))
        if isinstance(func, ast.Attribute) and func.attr == "strip" and not node.args:
            return evaluate(func.value, key, values).strip()
    raise ValueError("unsupported default expression")


def default_equal(getter, key, value):
    try:
        default = evaluate(getter.body, key, {})
        return evaluate(getter.body, key, {key: value}) == default, default
    except (TypeError, ValueError):
        return False, None


def read_command_defaults(fork):
    defaults = {}
    for filename, classname, names in [
        ("parallel", "ParallelConfig", {"pipeline_parallel_size", "tensor_parallel_size"}),
        ("scheduler", "SchedulerConfig", {"long_prefill_token_threshold"}),
    ]:
        source = git_source(fork, f"vllm/config/{filename}.py")
        cls = next(n for n in ast.parse(source).body
                   if isinstance(n, ast.ClassDef) and n.name == classname)
        for node in cls.body:
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            if node.target.id not in names:
                continue
            value = node.value
            if isinstance(value, ast.Call):
                value = next(k.value for k in value.keywords if k.arg == "default")
            defaults[node.target.id] = ast.literal_eval(value)
    if set(defaults) != {"pipeline_parallel_size", "tensor_parallel_size",
                         "long_prefill_token_threshold"}:
        raise ValueError("cannot prove omitted command defaults")
    return defaults


def dry_run(root, layout):
    with tempfile.TemporaryDirectory(prefix="effective-env-") as scratch:
        base = Path(scratch)
        (base / "bin").mkdir()
        (base / "bin/python").symlink_to(sys.executable)
        package = base / "vllm"
        package.mkdir()
        (package / "__init__.py").touch()
        (package / "_ampere_marlin_C.py").touch()
        capture = base / "exports"
        # Capture the same env call that DRY prints, before its display filter.
        wrapper = 'serve=$1; capture=$2; shift 2; env() { command env -0 > "$capture"; command env "$@"; }; source "$serve"'
        result = subprocess.run(
            ["bash", "-c", wrapper, "env-match", str(root / "serve.sh"), str(capture)],
            cwd=root, env={"PATH": os.defpath, "CUDA_VISIBLE_DEVICES": "",
                           "DRY": "1", "P2P": "force", "LAYOUT": layout,
                           "VENV": scratch, "PYTHONPATH": scratch},
            text=True, capture_output=True,
        )
        if result.returncode:
            raise ValueError(f"dry run failed ({result.returncode}): {result.stderr.strip()}")
        actual = dict(item.split("=", 1) for item in capture.read_text().split("\0") if item)
        # These are checker controls or shell bookkeeping, not recipe exports.
        for key in ("DRY", "P2P", "CUDA_VISIBLE_DEVICES", "_"):
            actual.pop(key, None)
        command = shlex.split(result.stdout.split("DRY=1, command:\n", 1)[1])
        return actual, parse_command(command), result.stdout


def parse_command(command):
    if len(command) < 3 or command[1] != "serve":
        raise ValueError("expected vllm serve command")
    values = {"model": command[2]}
    index = 3
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected command token {token!r}")
        key = token[2:].replace("-", "_")
        if key in values:
            raise ValueError(f"duplicate command argument {key}")
        if index + 1 == len(command) or command[index + 1].startswith("--"):
            value = True
        else:
            index += 1
            value = command[index]
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        values[key] = [value] if key == "served_model_name" else value
        index += 1
    return values


def log_command(path):
    commands = [ast.literal_eval(line.split("non-default args:", 1)[1].strip())
                for line in path.read_text().splitlines() if "non-default args:" in line]
    if not commands or any(command != commands[0] for command in commands):
        raise ValueError("serve.log must contain one consistent non-default args record")
    return commands[0]


def compare_commands(actual, expected, defaults):
    actual, expected = dict(actual), dict(expected)
    for key in sorted(IGNORED_ARGS):
        actual.pop(key, None)
        expected.pop(key, None)
        print(f"command: allowed {key} (model path, bind address, or tokens-details flag)")
    for values in (actual, expected):
        spec = values.get("speculative_config")
        if isinstance(spec, dict):
            values["speculative_config"] = {k: v for k, v in spec.items() if k != "model"}
    print("command: allowed speculative_config.model (drafter path)")
    for key, value in defaults.items():
        actual.setdefault(key, value)
        expected.setdefault(key, value)
    failures = []
    for key in sorted(set(actual) | set(expected)):
        if key not in actual or key not in expected or actual[key] != expected[key]:
            failures.append(f"command: {key}: validation={expected.get(key)!r}, recipe={actual.get(key)!r}")
        else:
            print(f"command: MATCH {key}={actual[key]!r}")
    return failures


def compare_environment(actual, expected, getters, source, layout):
    failures = []
    matched = allowed = 0
    for key in sorted(set(actual) | set(expected)):
        if key in actual and key in expected and actual[key] == expected[key]:
            print(f"environment: MATCH {key}" + (" (path)" if key in PATH_KEYS else f"={actual[key]!r}"))
            matched += 1
            continue
        reason = LAB_KEYS.get(key)
        if key in PATH_KEYS:
            reason = "installation, checkpoint, workspace, or executable search path"
        if key == "EXTRA_ARGS" and expected.get(key) == "--enable-prompt-tokens-details" and key not in actual:
            reason = "validation tokens-details flag; ignored in command comparison"
        if key == "VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE" and layout == "pp4":
            reason = "PP4 has TP=1 and no tensor-parallel all-reduce"
        # Missing exports can be equivalent only through the pinned fork getter.
        if reason is None and ((key in actual) != (key in expected)) and key in getters:
            value = actual[key] if key in actual else expected[key]
            equal, default = default_equal(getters[key], key, value)
            if equal:
                getter = getters[key]
                expression = " ".join(ast.get_source_segment(source, getter).split())
                proof = (f"explicit {value!r} equals fork default {default!r}; "
                         f"vllm/envs.py:{getter.lineno} at {PIN[:10]}: {expression}")
                if key not in actual:
                    # A validation-only default equals the recipe's effective value.
                    print(f"environment: MATCH {key} ({proof}; recipe inherits default)")
                    matched += 1
                    continue
                reason = proof
        if reason is not None:
            print(f"environment: allowed {key} ({reason})")
            allowed += 1
        else:
            failures.append(f"environment: {key}: validation={expected.get(key)!r}, recipe={actual.get(key)!r}")
    print(f"environment: {matched} matches (literal or proven default); {allowed} allowed differences; {len(failures)} mismatches")
    return failures


def check_effective(root, path, layout, fork, serve_log=None):
    expected = json.loads(path.read_text())
    if not isinstance(expected, dict) or not all(isinstance(v, str) for v in expected.values()):
        raise ValueError("effective environment must be a complete string-valued JSON object")
    source = git_source(fork, "vllm/envs.py")
    getters = read_getters(source)
    defaults = read_command_defaults(fork)
    actual, command, stdout = dry_run(root, layout)
    logged = log_command(serve_log or path.with_name("serve.log"))
    # Lab launcher aliases are checked against their resolved server settings.
    if "MAX_BATCHED" in expected:
        actual["MAX_BATCHED"] = str(command["max_num_batched_tokens"])
        print("environment: MAX_BATCHED resolved from --max-num-batched-tokens")
    if "GLM5_PP4_DRAFT_WIDTH" in expected:
        actual["GLM5_PP4_DRAFT_WIDTH"] = actual["VLLM_GLM5_DFLASH_ADAPTIVE_DRAFT_WIDTH"]
        print("environment: GLM5_PP4_DRAFT_WIDTH resolved from adaptive draft width")
    print(f"Full environment: layout={layout}; DRY=1; P2P=force; fork={PIN}")
    print("CPU fixture: optional Marlin module discoverable; no runtime imports")
    print("Capture excludes checker controls DRY/P2P/CUDA_VISIBLE_DEVICES and shell bookkeeping _")
    for line in stdout.splitlines():
        if "[release-features] recover=" in line:
            print(line)
    failures = compare_environment(actual, expected, getters, source, layout)
    command_failures = compare_commands(command, logged, defaults)
    print(f"command: {len(command_failures)} mismatches (omitted defaults read from pinned config files)")
    failures.extend(command_failures)
    if failures:
        print("\n".join(failures))
    print(f"Full environment match: {'FAIL' if failures else 'PASS'} ({layout}; {len(failures)} mismatches)")
    return int(bool(failures))
