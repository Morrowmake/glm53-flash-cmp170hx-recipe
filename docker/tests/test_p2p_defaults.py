from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_default_samples_use_content_gate_not_hard_off():
    assert "P2P=auto" in (ROOT / "docker/container.env").read_text()
    assert "\nVLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0\n" not in (ROOT / "docker/container.env").read_text()
    for path in (".env.example", "examples/tp4.env"):
        assert "# P2P=auto" in (ROOT / path).read_text()
    serve = (ROOT / "serve.sh").read_text()
    assert "VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE:-0" not in serve
    assert "PYTORCH_CUDA_ALLOC_CONF-expandable_segments:False" in serve
