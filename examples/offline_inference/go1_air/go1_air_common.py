#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for GO-1-Air offline inference.

All functions are free-standing. No internal model-core imports.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def tensor_dtype(name: str) -> torch.dtype:
    """Map a string name to a torch.dtype."""
    mapping: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower().replace("_", "")
    if key not in mapping:
        raise ValueError(
            f"Unknown dtype '{name}'. Expected one of: {sorted(set(v for v in mapping if v == key or name in v))}"
        )
    return mapping[key]


def make_shared_noise(
    seed: int,
    sample_index: int,
    shape: tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministic noise tensor seeded by ``seed + sample_index``."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + sample_index)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    return noise.to(device=device, dtype=dtype)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 hex digest of a contiguous CPU copy of *tensor*."""
    arr = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def save_actions_npy(
    output_dir: Path,
    actions: torch.Tensor,
    filename: str = "actions.npy",
) -> Path:
    """Save *actions* as a .npy file under *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    np.save(path, actions.detach().contiguous().cpu().numpy())
    return path


def save_json(path: Path, payload: dict) -> Path:
    """Write *payload* as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False, default=_default)
    return path


def capture_env_snapshot(output_dir: Path) -> Path:
    """Write ``env.json`` with version and system information.

    Uses ``importlib.metadata.version()`` so that we never trigger a
    ``vllm._C`` / CUDA import just for version strings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "env.json"

    def _pkg_version(name: str) -> str | None:
        try:
            from importlib.metadata import version

            return version(name)
        except Exception:
            return None

    def _git_commit(repo_root: Path | None = None) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(repo_root) if repo_root else None,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    snapshot = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "cuda_available": torch.cuda.is_available(),
        "transformers_version": _pkg_version("transformers"),
        "safetensors_version": _pkg_version("safetensors"),
        "vllm_version": _pkg_version("vllm"),
        "vllm_omni_version": _pkg_version("vllm_omni"),
        "git_commit": _git_commit(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True, ensure_ascii=False)
    return path


def capture_gpu_snapshot(output_dir: Path) -> Path:
    """Write ``gpu.json`` with nvidia-smi output, or ``available=false``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gpu.json"

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise FileNotFoundError("nvidia-smi returned non-zero")
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mib": parts[2],
                    "memory_used_mib": parts[3],
                    "memory_free_mib": parts[4],
                    "utilization_gpu_pct": parts[5],
                })
        snapshot = {"available": True, "gpus": gpus}
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        snapshot = {"available": False, "reason": str(exc)}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True, ensure_ascii=False)
    return path


def summarize_actions(actions: torch.Tensor) -> dict:
    """Return a summary dict of the action tensor."""
    ac = actions.detach().float().cpu()
    return {
        "shape": list(ac.shape),
        "dtype": str(ac.dtype).split(".")[-1],
        "device": str(actions.device),
        "min": float(ac.min().item()),
        "max": float(ac.max().item()),
        "mean": float(ac.mean().item()),
        "std": float(ac.std().item()) if ac.numel() > 0 else 0.0,
        "has_nan": bool(torch.isnan(ac).any().item()),
        "has_inf": bool(torch.isinf(ac).any().item()),
        "all_zero": bool((ac == 0.0).all().item()),
        "sha256": tensor_sha256(actions),
    }


def compare_with_upstream(
    vllm_actions: torch.Tensor,
    upstream_path: Path,
) -> dict:
    """Compare vLLM actions against a saved upstream tensor.

    Supports ``.npy`` (preferred) and ``.pt`` / ``.pth`` (loaded with
    ``map_location="cpu", weights_only=True``).
    """
    vllm_cpu = vllm_actions.detach().float().cpu()
    suffix = upstream_path.suffix.lower()

    if suffix == ".npy":
        upstream = torch.from_numpy(np.load(upstream_path)).float()
    elif suffix in (".pt", ".pth"):
        upstream = torch.load(upstream_path, map_location="cpu", weights_only=True)
        if not isinstance(upstream, torch.Tensor):
            raise TypeError(f"Upstream file contains {type(upstream)}, expected a torch.Tensor.")
        upstream = upstream.float()
    else:
        raise ValueError(f"Unsupported upstream file format: {suffix}. Expected .npy, .pt, or .pth.")

    if vllm_cpu.shape != upstream.shape:
        return {
            "error": f"Shape mismatch: vllm={list(vllm_cpu.shape)} upstream={list(upstream.shape)}",
            "vllm_shape": list(vllm_cpu.shape),
            "upstream_shape": list(upstream.shape),
        }

    diff = vllm_cpu - upstream
    mse = float((diff ** 2).mean().item())
    return {
        "vllm_shape": list(vllm_cpu.shape),
        "upstream_shape": list(upstream.shape),
        "vllm_dtype": str(vllm_actions.dtype).split(".")[-1],
        "upstream_dtype": str(upstream.dtype).split(".")[-1],
        "max_abs_error": float(diff.abs().max().item()),
        "mean_abs_error": float(diff.abs().mean().item()),
        "mse": mse,
        "sha256_vllm": tensor_sha256(vllm_actions),
        "sha256_upstream": tensor_sha256(upstream),
    }


def sync_cuda_if_needed(device: torch.device | str) -> None:
    """Synchronize the CUDA stream if *device* is a CUDA device."""
    device_str = str(device)
    if "cuda" in device_str and torch.cuda.is_available():
        torch.cuda.synchronize()
