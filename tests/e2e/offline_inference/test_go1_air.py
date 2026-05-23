# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline E2E test for GO-1-Air inference.

Provides two test functions:

* ``test_go1_air_offline_cpu_stub`` — always runs (core_model): exercises
  the end2end harness on CPU with a fake/stub batch.  No GPU, no
  checkpoint, no dataset required.

* ``test_go1_air_offline_real`` — gated behind ``advanced_model`` /
  ``hardware_test``: runs with a real checkpoint when ``GO1_AIR_MODEL_DIR``
  is set and CUDA is available.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_SCRIPT = REPO_ROOT / "examples" / "offline_inference" / "go1_air" / "end2end.py"


def _run_end2end(output_dir: Path, extra_args: list[str] | None = None) -> None:
    cmd = [
        sys.executable,
        str(EXAMPLE_SCRIPT),
        "--device", "cpu",
        "--dtype", "float32",
        "--num-samples", "1",
        "--output-dir", str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CPU stub test — always eligible (no GPU, no checkpoint, no dataset).
# ---------------------------------------------------------------------------

@pytest.mark.diffusion
@pytest.mark.core_model
def test_go1_air_offline_cpu_stub() -> None:
    """Run end2end.py on CPU with a fake batch and verify output artifacts."""
    with tempfile.TemporaryDirectory(prefix="go1_air_e2e_cpu_") as tmpdir:
        output_dir = Path(tmpdir) / "outputs"
        _run_end2end(output_dir)

        # Required artifacts
        for filename in ["actions.npy", "metadata.json", "env.json", "gpu.json"]:
            p = output_dir / filename
            assert p.exists(), f"Missing artifact: {p}"

        # Validate metadata.json
        with open(output_dir / "metadata.json", encoding="utf-8") as f:
            md = json.load(f)

        assert md.get("device") == "cpu", f"Expected cpu device, got {md.get('device')}"

        summary = md.get("action_summary") or {}
        shape = summary.get("shape")
        assert shape == [1, 30, 16], f"Expected [1, 30, 16] action shape, got {shape}"
        assert summary.get("has_nan") is False, "Action tensor contains NaN"
        assert summary.get("has_inf") is False, "Action tensor contains Inf"

        chunk_size = md.get("chunk_size")
        max_action_dim = md.get("max_action_dim")
        assert chunk_size == 30, f"Expected chunk_size 30, got {chunk_size}"
        assert max_action_dim == 16, f"Expected max_action_dim 16, got {max_action_dim}"


# ---------------------------------------------------------------------------
# Real-checkpoint test — gated: advanced_model + H100/MI325 + env var.
# ---------------------------------------------------------------------------

@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_go1_air_offline_real(run_level: str) -> None:
    """Run end2end.py with a real GO-1-Air checkpoint.

    Skipped unless ``GO1_AIR_MODEL_DIR`` is set, ``run_level`` is
    ``advanced_model``, and a supported GPU is present.
    """
    if run_level != "advanced_model":
        pytest.skip("GO-1-Air real-checkpoint test requires --run-level advanced_model.")

    model_dir = os.getenv("GO1_AIR_MODEL_DIR")
    if not model_dir:
        pytest.skip("GO1_AIR_MODEL_DIR is not set")

    if not os.path.isdir(model_dir):
        pytest.skip(f"GO1_AIR_MODEL_DIR is not a directory: {model_dir}")

    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass

    if not cuda_available:
        pytest.skip("CUDA is not available; GO-1-Air real-checkpoint test requires GPU.")

    with tempfile.TemporaryDirectory(prefix="go1_air_e2e_real_") as tmpdir:
        output_dir = Path(tmpdir) / "outputs"
        extra_args = [
            "--model-dir", model_dir,
            "--device", "cuda",
            "--dtype", "bfloat16",
            "--num-samples", "1",
        ]
        _run_end2end(output_dir, extra_args=extra_args)

        # Required artifacts
        for filename in ["actions.npy", "metadata.json", "env.json", "gpu.json"]:
            p = output_dir / filename
            assert p.exists(), f"Missing artifact: {p}"

        # Validate metadata.json
        with open(output_dir / "metadata.json", encoding="utf-8") as f:
            md = json.load(f)

        summary = md.get("action_summary") or {}
        shape = summary.get("shape")
        assert shape == [1, 30, 16], f"Expected [1, 30, 16] action shape, got {shape}"
        assert summary.get("has_nan") is False, "Action tensor contains NaN"
        assert summary.get("has_inf") is False, "Action tensor contains Inf"

        assert md.get("runtime_mode") == "real_checkpoint_loaded", (
            f"Expected real_checkpoint_loaded, got {md.get('runtime_mode')}"
        )
