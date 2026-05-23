#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GO-1-Air end-to-end offline inference harness.

Runs one or more fake/stub samples through the GO-1-Air pipeline, saves
artifacts (actions.npy, metadata.json, env.json, gpu.json), and optionally
compares against a user-provided upstream action tensor.

Two modes:

* **Stub** — ``--model-dir`` is absent and ``--stub-ok`` is true (the
  default).  A no-checkpoint ``Go1AirPipeline`` is built so the plumbing
  can be exercised end-to-end (zero actions returned).
* **Real** — ``--model-dir`` points at a checkpoint directory containing
  ``config.json`` and ``model.safetensors[.index.json]``.  Weights are
  loaded and real actions are produced.

If ``--model-dir`` is provided but the directory lacks safetensors, the
script exits with an error (no silent fallback to stub mode).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm_omni.diffusion.data import OmniDiffusionConfig  # noqa: E402
from vllm_omni.diffusion.models.go1_air import Go1AirPipeline  # noqa: E402
from vllm_omni.diffusion.models.go1_air.config import (  # noqa: E402
    OBS_IMAGES,
    OBS_STATE,
    OBS_TASK,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402

from go1_air_common import (  # noqa: E402
    capture_env_snapshot,
    capture_gpu_snapshot,
    compare_with_upstream,
    make_shared_noise,
    save_actions_npy,
    save_json,
    summarize_actions,
    sync_cuda_if_needed,
    tensor_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GO-1-Air end-to-end offline inference.")
    parser.add_argument(
        "--model-dir",
        default=os.getenv("GO1_AIR_MODEL_DIR", ""),
        help="Path to a GO-1-Air checkpoint directory (optional for stub mode).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of fake samples to run (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed (default: 1234).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16" if torch.cuda.is_available() else "float32",
        choices=["bfloat16", "float32"],
        help="Torch dtype for the pipeline (default: bfloat16 on CUDA, float32 on CPU).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend (default: eager).",
    )
    parser.add_argument(
        "--enable-warmup",
        action="store_true",
        help="Run a pipeline warmup forward before the timed pass.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Fail if any checkpoint keys are missing or unexpected.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/go1_air/vllm_infer",
        help="Directory for output artifacts (default: outputs/go1_air/vllm_infer).",
    )
    parser.add_argument(
        "--benchmark-forward",
        action="store_true",
        help="Run warmup + benchmark iterations and save forward_latency.json.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Warmup iterations for benchmark mode (default: 3).",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=10,
        help="Benchmark iterations for benchmark mode (default: 10).",
    )
    parser.add_argument(
        "--upstream-action-path",
        default=None,
        help="Path to a saved upstream action tensor (.npy or .pt) for comparison.",
    )
    parser.add_argument(
        "--stub-ok",
        action="store_true",
        default=True,
        help="Allow stub/no-checkpoint mode when --model-dir is absent (default: true).",
    )
    parser.add_argument(
        "--no-stub-ok",
        action="store_false",
        dest="stub_ok",
        help="Disallow stub mode; require a real checkpoint.",
    )
    parser.add_argument(
        "--fail-on-all-zero",
        action="store_true",
        help="Treat an all-zero action output as a failure.",
    )
    return parser.parse_args()


def build_fake_batch(*, image_size: int, state_dim: int, device: str, dtype: torch.dtype) -> dict:
    """Mirrors ``examples/offline_inference/go1_air/smoke.py:build_fake_batch``."""
    return {
        OBS_STATE: torch.zeros((1, state_dim), device=device, dtype=dtype),
        OBS_TASK: ["pick up the red block"],
        f"{OBS_IMAGES}.image0": torch.zeros(
            (1, 1, 3, image_size, image_size),
            device=device,
            dtype=dtype,
        ),
        f"{OBS_IMAGES}.image0_mask": torch.ones((1,), device=device, dtype=torch.bool),
    }


def _latency_summary(values_ms: list[float]) -> dict:
    if not values_ms:
        return {}
    sorted_vals = sorted(values_ms)
    p50_idx = min(len(sorted_vals) - 1, round(0.50 * (len(sorted_vals) - 1)))
    p90_idx = min(len(sorted_vals) - 1, round(0.90 * (len(sorted_vals) - 1)))
    import statistics
    return {
        "mean_ms": float(statistics.mean(sorted_vals)),
        "min_ms": float(sorted_vals[0]),
        "max_ms": float(sorted_vals[-1]),
        "p50_ms": float(sorted_vals[p50_idx]),
        "p90_ms": float(sorted_vals[p90_idx]),
    }


def run_pipeline_forward(pipeline, batch_inputs: dict, noise: torch.Tensor | None) -> torch.Tensor:
    """Run one forward pass through the pipeline.  Returns the action tensor."""
    sampling_params = OmniDiffusionSamplingParams(
        extra_args={
            "batch_inputs": batch_inputs,
            "noise": noise,
        }
    )
    req = OmniDiffusionRequest(prompts=[""], sampling_params=sampling_params)
    output = pipeline.forward(req)
    if output.error:
        raise RuntimeError(f"Go1AirPipeline.forward returned error: {output.error}")
    if output.output is None:
        raise RuntimeError("Go1AirPipeline.forward returned no output tensor.")
    return output.output


def main() -> int:
    args = parse_args()

    model_dir_provided = bool(args.model_dir and args.model_dir.strip())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Capture environment early (before touching any vllm CUDA internals).
    _ = capture_env_snapshot(output_dir)
    _ = capture_gpu_snapshot(output_dir)

    torch_dtype = tensor_dtype(args.dtype)

    custom_pipeline_args: dict = {
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "strict_load": args.strict_load,
    }
    if args.enable_warmup:
        custom_pipeline_args["enable_warmup"] = True

    od_config = OmniDiffusionConfig(
        model_class_name="Go1AirPipeline",
        model=args.model_dir or "",
        dtype=args.dtype,
        custom_pipeline_args=custom_pipeline_args,
    )

    pipeline = Go1AirPipeline(od_config=od_config)
    cfg = pipeline.config

    print(f"[go1_air] runtime_mode={pipeline.runtime_mode()}")
    print(f"[go1_air] action chunk={cfg.chunk_size} action_dim={cfg.max_action_dim}")

    # Enforce: if --model-dir was given, a real checkpoint must be loaded.
    if model_dir_provided and not pipeline.has_real_checkpoint():
        print(
            f"[go1_air] FAIL: --model-dir={args.model_dir} was provided but no",
            "safetensors checkpoint was found under that path.",
            "Refusing to silently fall back to stub mode.",
        )
        return 1

    # Enforce: stub mode only when allowed.
    if not model_dir_provided:
        if not args.stub_ok:
            print(
                "[go1_air] FAIL: --model-dir is absent and --stub-ok is disabled.",
                "Provide --model-dir or pass --stub-ok.",
            )
            return 1
        print("[go1_air] running in stub/no-checkpoint mode")

    # Build the fake batch (mirrors smoke.py exactly).
    batch = build_fake_batch(
        image_size=cfg.image_resolution[0],
        state_dim=cfg.max_state_dim,
        device=cfg.device,
        dtype=torch_dtype,
    )

    all_summaries: list[dict] = []
    timing_ms: list[float] = []

    for sample_idx in range(args.num_samples):
        noise = make_shared_noise(
            seed=args.seed,
            sample_index=sample_idx,
            shape=(1, cfg.chunk_size, cfg.max_action_dim),
            device=cfg.device,
            dtype=torch_dtype,
        )

        sync_cuda_if_needed(cfg.device)
        t_start = time.perf_counter()
        actions = run_pipeline_forward(pipeline, batch, noise)
        sync_cuda_if_needed(cfg.device)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        timing_ms.append(elapsed_ms)

        summary = summarize_actions(actions)
        summary["sample_index"] = sample_idx
        summary["elapsed_ms"] = elapsed_ms
        all_summaries.append(summary)

        print(f"[go1_air] sample {sample_idx} shape={tuple(actions.shape)} elapsed={elapsed_ms:.1f}ms")

        if actions.shape != (1, cfg.chunk_size, cfg.max_action_dim):
            print(
                f"[go1_air] FAIL: expected shape (1, {cfg.chunk_size}, {cfg.max_action_dim}), "
                f"got {tuple(actions.shape)}"
            )
            return 1

        if args.fail_on_all_zero and summary["all_zero"]:
            print("[go1_air] FAIL: action output is all-zero and --fail-on-all-zero is set.")
            return 1

    # --- save artifacts ---
    final_actions = run_pipeline_forward(
        pipeline,
        batch,
        make_shared_noise(
            seed=args.seed,
            sample_index=0,
            shape=(1, cfg.chunk_size, cfg.max_action_dim),
            device=cfg.device,
            dtype=torch_dtype,
        ),
    )

    actions_path = save_actions_npy(output_dir, final_actions)
    print(f"[go1_air] actions saved to {actions_path}")

    metadata = {
        "runtime_mode": pipeline.runtime_mode(),
        "model_dir": str(Path(args.model_dir).resolve()) if model_dir_provided else None,
        "model_class": "Go1AirPipeline",
        "device": cfg.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "chunk_size": cfg.chunk_size,
        "max_action_dim": cfg.max_action_dim,
        "action_summary": all_summaries[0] if len(all_summaries) == 1 else None,
        "all_summaries": all_summaries if len(all_summaries) > 1 else None,
        "timing": {
            "samples": timing_ms,
            "total_ms": sum(timing_ms),
        },
        "argv": sys.argv,
        "output_files": {
            "actions": str(actions_path),
            "metadata": str(output_dir / "metadata.json"),
            "env": str(output_dir / "env.json"),
            "gpu": str(output_dir / "gpu.json"),
        },
    }
    metadata_path = save_json(output_dir / "metadata.json", metadata)
    print(f"[go1_air] metadata saved to {metadata_path}")

    # --- optional benchmark ---
    if args.benchmark_forward:
        noise0 = make_shared_noise(
            seed=args.seed,
            sample_index=0,
            shape=(1, cfg.chunk_size, cfg.max_action_dim),
            device=cfg.device,
            dtype=torch_dtype,
        )

        warmup_ms: list[float] = []
        for _ in range(args.warmup_iters):
            sync_cuda_if_needed(cfg.device)
            t0 = time.perf_counter()
            _ = run_pipeline_forward(pipeline, batch, noise0)
            sync_cuda_if_needed(cfg.device)
            warmup_ms.append((time.perf_counter() - t0) * 1000.0)

        bench_ms: list[float] = []
        for _ in range(args.benchmark_iters):
            sync_cuda_if_needed(cfg.device)
            t0 = time.perf_counter()
            _ = run_pipeline_forward(pipeline, batch, noise0)
            sync_cuda_if_needed(cfg.device)
            bench_ms.append((time.perf_counter() - t0) * 1000.0)

        latency_json = {
            "warmup_iters": args.warmup_iters,
            "benchmark_iters": args.benchmark_iters,
            "warmup_summary": _latency_summary(warmup_ms),
            "benchmark_summary": _latency_summary(bench_ms),
            "benchmark_samples_ms": bench_ms,
        }
        latency_path = save_json(output_dir / "forward_latency.json", latency_json)
        print(f"[go1_air] latency summary saved to {latency_path}")
        print(f"[go1_air] benchmark mean={latency_json['benchmark_summary'].get('mean_ms', 0):.1f}ms")

    # --- optional upstream comparison ---
    if args.upstream_action_path:
        upstream_path = Path(args.upstream_action_path)
        if not upstream_path.exists():
            print(f"[go1_air] WARNING: upstream action path not found: {upstream_path}")
        else:
            comparison = compare_with_upstream(final_actions, upstream_path)
            comparison_path = save_json(output_dir / "comparison.json", comparison)
            print(f"[go1_air] comparison saved to {comparison_path}")
            if "error" not in comparison:
                print(
                    f"[go1_air] upstream comparison: "
                    f"max_abs_error={comparison['max_abs_error']:.6f} "
                    f"mean_abs_error={comparison['mean_abs_error']:.6f}"
                )

    print(f"[go1_air] OK action shape={tuple(final_actions.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
