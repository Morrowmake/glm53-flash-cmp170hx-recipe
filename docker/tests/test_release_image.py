import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import cache_seed as cache
import oci_layers as oci
import payload

PIN = "a" * 40

def test_partition():
    cases = {"python/bin/python": "python", "venv/lib/site-packages/torch/x.py": "dependencies",
             "venv/lib/site-packages/vllm-1.dist-info/METADATA": "engine",
             "venv/lib/site-packages/__editable__.vllm-1.pth": "engine",
             "venv/lib/site-packages/__editable___vllm_1_finder.py": "engine",
             "venv/.ampere-marlin-image-stamp": "engine", "venv/bin/vllm": "engine",
             "vllm-src/vllm/x.py": "engine", "vllm-src/vllm/x.abi3.so": "native"}
    for path, part in cases.items():
        assert payload.partition(Path(path)) == part

@pytest.mark.parametrize("version", ["1.2.3", "1.2.4"])
def test_lock(tmp_path, version):
    opt = tmp_path / "opt"
    path = opt / "venv/lib/python3.12/site-packages/python_dotenv-1.dist-info"
    path.mkdir(parents=True)
    (path / "METADATA").write_text("Name: python-dotenv\nVersion: " + version)
    constraints = tmp_path / "CONSTRAINTS"
    constraints.write_text("python-dotenv==1.2.3\n")
    if version == "1.2.3":
        payload.check_lock(opt, constraints)
    else:
        with pytest.raises(ValueError, match="lock mismatch"):
            payload.check_lock(opt, constraints)

def identity(tmp_path):
    lock = tmp_path / "lock"
    lock.write_text("python-dotenv==1.2.3")
    value = dict(schema=1, engine_commit=PIN, gpu_arch="sm_80", gpu_name="NVIDIA CMP 170HX",
                 driver=1, dependencies=cache.sha(lock), versions={"triton": "3.7.1"})
    path = tmp_path / "identity"
    path.write_text(json.dumps(value))
    return value, path, lock

def bundle(tmp_path):
    value, identity_path, lock = identity(tmp_path)
    source = tmp_path / "compiled"
    (source / ".triton/cache/key").mkdir(parents=True)
    (source / ".triton/cache/key/kernel.cubin").write_bytes(b"compiled")
    result = tmp_path / "bundle"
    cache.pack(source, identity_path, result)
    return value, lock, result

def test_bundle_integrity(tmp_path):
    value, lock, result = bundle(tmp_path)
    assert cache.check(result, PIN, lock)["identity"] == value
    with pytest.raises(ValueError, match="identity mismatch"):
        cache.check(result, "b" * 40, lock)
    with (result / "cache.tar.gz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        cache.check(result, PIN, lock)

@pytest.mark.parametrize("name", ["/outside", "../outside", ".triton/cache/../../outside", "huggingface/token"])
def test_reject_path(name):
    assert not cache.allowed(name)

def test_pack_reject_symlink(tmp_path):
    value, identity_path, lock = identity(tmp_path)
    source = tmp_path / "compiled/.triton/cache"
    source.mkdir(parents=True)
    (source / "escape").symlink_to(lock)
    with pytest.raises(ValueError, match="symlink"):
        cache.pack(tmp_path / "compiled", identity_path, tmp_path / "bundle")

def test_seed_mismatch_and_existing(tmp_path, monkeypatch):
    value, lock, result = bundle(tmp_path)
    opt = tmp_path / "opt/venv"
    opt.mkdir(parents=True)
    (opt / "requirements.lock.txt").write_bytes(lock.read_bytes())
    monkeypatch.setattr(cache, "runtime_identity", lambda opt: value)
    dest = tmp_path / "volume"
    exports = cache.seed(result, opt.parent, dest)
    root = dest / "compiled" / cache.key(value)
    file = root / ".triton/cache/key/kernel.cubin"
    assert file.read_bytes() == b"compiled" and "/cache/compiled/" in exports
    file.write_bytes(b"newer")
    cache.seed(result, opt.parent, dest)
    assert file.read_bytes() == b"newer"
    value["gpu_arch"] = "sm_90"
    cache.seed(result, opt.parent, dest)
    assert not (dest / "compiled" / cache.key(value) / ".seed-complete").exists()

def test_keys(tmp_path):
    value, path, lock = identity(tmp_path)
    k = cache.key(value)
    for name, replacement in (("engine_commit", "b"*40), ("gpu_arch", "sm_90"),
                              ("driver", 2), ("dependencies", "different")):
        assert cache.key(dict(value, **{name: replacement})) != k

def base(tmp_path):
    b = tmp_path / "base"
    (b / "blobs/sha256").mkdir(parents=True)
    c = dict(architecture="amd64", os="linux", config={}, rootfs=dict(type="layers", diff_ids=[]))
    desc = oci.put(b, oci.encoded(c), oci.OCI+"config.v1+json")
    m = dict(schemaVersion=2, config=desc, layers=[], mediaType=oci.OCI+"manifest.v1+json")
    md = oci.put(b, oci.encoded(m), m["mediaType"])
    (b / "index.json").write_text(json.dumps(dict(manifests=[md])))
    return b

def test_layer_reuse_and_reproducibility(tmp_path):
    b = base(tmp_path)
    p = tmp_path / "payload"
    for name in payload.PARTS:
        d = p / name / "opt" / name
        d.mkdir(parents=True)
        (d / "file").write_text(name)
    reports = []
    for n in range(3):
        out = tmp_path / str(n) / "oci"
        if n == 2:
            (p / "engine/opt/engine/file").write_text("changed Python")
        oci.assemble(b, p, out, PIN if n < 2 else "b"*40, "test", threads=2)
        reports.append(json.loads((out.parent / "layers.json").read_text()))
    assert reports[0] == reports[1]
    assert reports[0][:3] == reports[2][:3]
    assert reports[0][3]["digest"] != reports[2][3]["digest"]
    for out in (tmp_path / "0/oci", tmp_path / "2/oci"):
        m, c = oci.image(out)
        assert len(m["layers"]) == len(c["rootfs"]["diff_ids"]) == 4
        for desc, diff in zip(m["layers"], c["rootfs"]["diff_ids"]):
            import gzip
            data = gzip.decompress((out / "blobs/sha256" / desc["digest"].split(":")[1]).read_bytes())
            assert "sha256:" + hashlib.sha256(data).hexdigest() == diff

def test_kill_switches(tmp_path):
    b = base(tmp_path)
    p = tmp_path / "payload"
    for name in payload.PARTS:
        d = p / name
        d.mkdir(parents=True)
        (d / name).write_text(name)
    oci.assemble(b, p, tmp_path / "out/oci", PIN, "test", layered=False, parallel=False)
    m, c = oci.image(tmp_path / "out/oci")
    assert len(m["layers"]) == 1


def test_push_guards_and_private_auth(tmp_path):
    import os
    import subprocess
    b = base(tmp_path)
    digest = json.loads((b/"index.json").read_text())["manifests"][0]["digest"]
    record = tmp_path/"acceptance.json"
    record.write_text(json.dumps(dict(image_digest=digest, passed=True, artifact_audit_passed=True)))
    log = tmp_path/"calls"
    fake = tmp_path/"crane"
    fake.write_text("#!/bin/bash\nset -eu\nprintf '%s\\n' \"$*\" >> \"$CALL_LOG\"\ncase \"$1\" in\n auth) cat >/dev/null; echo \"$DOCKER_CONFIG\" > \"$CONFIG_LOG\" ;;\n push) ;;\n digest) echo \"$EXPECTED\" ;;\nesac\n")
    fake.chmod(0o755)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CRANE=str(fake), CALL_LOG=str(log),
               CONFIG_LOG=str(tmp_path/"config-path"), EXPECTED=digest)
    argv = ["bash", str(ROOT/"push.sh"), str(b), "ghcr.io/morrowmake/vllm-cmp170hx:test", digest, str(record)]
    assert subprocess.run(argv, env=env, input="", text=True, capture_output=True).returncode == 2
    assert not log.exists()
    env["IMAGE_PUBLISH"] = "1"
    secret = "fixture-private-auth-value"
    run = subprocess.run(argv, env=env, input=secret, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert secret not in run.stdout + run.stderr + log.read_text()
    assert not Path((tmp_path/"config-path").read_text().strip()).exists()
    record.write_text(json.dumps(dict(image_digest="sha256:"+"b"*64, passed=True, artifact_audit_passed=True)))
    log.unlink()
    assert subprocess.run(argv, env=env, input=secret, text=True, capture_output=True).returncode != 0
    assert not log.exists()

@pytest.mark.parametrize("enabled", ["0", "1"])
def test_launcher_dry_no_cuda(tmp_path, enabled):
    import os
    import subprocess
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", DRY="1", VLLM_IMAGE_CACHE_SEED=enabled,
               VLLM_GLM5_MARLIN_DECODE_CUDA="0",
               VENV=str(tmp_path/"missing"), MODEL=str(tmp_path/"target"), DFLASH_MODEL=str(tmp_path/"draft"))
    result = subprocess.run(["bash", str(ROOT.parent/"serve.sh")], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert ("[image-cache] disabled" if enabled == "0" else "[image-cache] enabled (dry run)") in result.stdout


def test_audit_token_shapes_and_nested_archive():
    import gzip
    import io
    from audit_oci import TOKEN_PATTERNS
    from runtime_tree import scan_stream
    scan_stream(io.BytesIO(b"ghp_example"), [], "fixture", credential_patterns=TOKEN_PATTERNS)
    for data in (b"ghp_" + b"x"*36, gzip.compress(b"github_pat_" + b"x"*50)):
        with pytest.raises(ValueError, match="Credential candidate"):
            scan_stream(io.BytesIO(data), [], "fixture", credential_patterns=TOKEN_PATTERNS)


def test_audit_collects_integrity_failure(tmp_path):
    from audit_oci import checked_layer
    root = tmp_path/"oci"
    (root/"blobs/sha256").mkdir(parents=True)
    name = "a"*64
    (root/"blobs/sha256"/name).write_bytes(b"wrong bytes")
    desc = dict(digest="sha256:"+name, size=11)
    result = checked_layer((root, desc, "sha256:"+name, [], []))
    assert result["passed"] is False and "descriptor mismatch" in result["reason"]


@pytest.mark.parametrize("name", [".cache/torch_extensions/kernel/module.so", ".nv/ComputeCache/kernel"])
def test_cpp_and_driver_caches(name):
    assert cache.allowed(name)


def test_audit_ignores_embedded_symbol_and_hash_fragments():
    import io
    from audit_oci import TOKEN_PATTERNS
    from runtime_tree import scan_stream
    scan_stream(io.BytesIO(b"symbol_ghr_" + b"x"*50), [], "fixture", credential_patterns=TOKEN_PATTERNS)


def test_review_bound_to_exact_bytes_and_offset(monkeypatch):
    import io
    from runtime_tree import scan_stream
    value = b"prefix example suffix"
    digest = hashlib.sha256(value).hexdigest()
    monkeypatch.setenv("IMAGE_REVIEWED_BYTE_MATCHES", json.dumps({digest: [7]}))
    scan_stream(io.BytesIO(value), [b"example"], "fixture")
    with pytest.raises(ValueError, match="Forbidden identifier"):
        scan_stream(io.BytesIO(value + b"changed"), [b"example"], "fixture")
    monkeypatch.setenv("IMAGE_REVIEWED_BYTE_MATCHES", json.dumps({digest: [8]}))
    with pytest.raises(ValueError, match="Forbidden identifier"):
        scan_stream(io.BytesIO(value), [b"example"], "fixture")


def test_review_cannot_allow_credential_candidates(monkeypatch):
    import io
    from audit_oci import TOKEN_PATTERNS
    from runtime_tree import scan_stream
    value = b"ghp_" + b"x"*36
    digest = hashlib.sha256(value).hexdigest()
    monkeypatch.setenv("IMAGE_REVIEWED_BYTE_MATCHES", json.dumps({digest: [0]}))
    with pytest.raises(ValueError, match="Credential candidate"):
        scan_stream(io.BytesIO(value), [b"ghp_"], "fixture", credential_patterns=TOKEN_PATTERNS)
