#!/usr/bin/env python3
"""Check local OCI integrity and scan every distributed layer and config."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path
import sys

from oci_layers import digest_file, image
from runtime_tree import scan_stream

# Candidates are reported without including their value.
TOKEN_PATTERNS = [re.compile(rb"github_pat_[a-z0-9_]{40,}"), re.compile(rb"gh[pousr]_[a-z0-9]{30,}")]

def layer_check(job):
    root, desc, diff, forbidden, patterns = job
    path = root / "blobs/sha256" / desc["digest"].split(":")[1]
    if digest_file(path) != desc["digest"] or path.stat().st_size != desc["size"]:
        raise ValueError("Layer descriptor mismatch")
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    if "sha256:" + h.hexdigest() != diff:
        raise ValueError("Layer diff ID mismatch")
    with path.open("rb") as f:
        scan_stream(f, forbidden, desc["digest"], credential_patterns=patterns)
    return desc["digest"]

def checked_layer(job):
    try:
        return dict(digest=layer_check(job), passed=True)
    except ValueError as error:
        return dict(digest=job[1]["digest"], passed=False, reason=str(error))

def audit(root, workers=8):
    manifest, config = image(root)
    forbidden = [s.lower().encode() for s in json.loads(os.environ.get("IMAGE_FORBIDDEN_STRINGS", "[]"))]
    patterns = TOKEN_PATTERNS + [re.compile(p.encode()) for p in
                  json.loads(os.environ.get("IMAGE_FORBIDDEN_PATTERNS", "[]"))]
    metadata = json.dumps(dict(manifest=manifest, config=config)).encode()
    scan_stream(io.BytesIO(metadata), forbidden, "OCI metadata", credential_patterns=patterns)
    if len(manifest["layers"]) != len(config["rootfs"]["diff_ids"]):
        raise ValueError("Layer count mismatch")
    jobs = [(root, d, diff, forbidden, patterns) for d, diff in
            zip(manifest["layers"], config["rootfs"]["diff_ids"])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        checked = list(pool.map(checked_layer, jobs))
    record = dict(passed=all(r["passed"] for r in checked), layers=len(checked), checked=checked,
                  formats=["OCI JSON", "gzip", "bzip2", "xz", "tar", "zip"],
                  gaps=["Embedded ELF fatbins and nonstandard compression (including zstd) require separate inspection",
                        "Credential candidates use known registry-token shapes"])
    print(json.dumps(record, sort_keys=True))
    return record

if __name__ == "__main__":
    result = audit(Path(sys.argv[1]), int(os.environ.get("IMAGE_AUDIT_WORKERS", "8")))
    sys.exit(0 if result["passed"] else 1)
