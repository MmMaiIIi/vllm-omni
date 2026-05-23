# GO-1-Air

> Source repository: <https://github.com/OpenDriveLab/AgiBot-World>
>
> Weights: <https://huggingface.co/agibot-world/GO-1-Air>

This example runs offline inference for **AgiBot GO-1-Air**, an open-source
Vision-Language-Latent-Action (ViLLA) policy. The released checkpoint ships
with the Latent Planner disabled, so this integration covers the
`Vision → Language → diffusion action` path only; an `[B, 30, 16]` action
chunk is produced via a 5-step squared-cosine flow-matching schedule.

## Quick start

```bash
# (Optional) point at a checkpoint directory containing config.json
# and model.safetensors[.index.json].
export GO1_AIR_MODEL_DIR=/path/to/GO-1-Air

bash examples/offline_inference/go1_air/run.sh
```

The smoke test prints `[smoke] OK action shape=(1, 30, 16)` on success.

## End-to-end evaluation

The full end-to-end harness (`end2end.py`) runs one or more samples through the
GO-1-Air pipeline and saves structured artifacts:

```bash
# CPU stub mode (no checkpoint required):
bash examples/offline_inference/go1_air/run.sh --end2end \
    --device cpu --dtype float32 --num-samples 1

# With a real checkpoint (requires GO1_AIR_MODEL_DIR):
export GO1_AIR_MODEL_DIR=/path/to/GO-1-Air
bash examples/offline_inference/go1_air/run.sh --end2end \
    --device cuda --dtype bfloat16 --num-samples 1
```

## Result collection

`collect_results.sh` wraps `end2end.py` with environment snapshots, GPU
monitoring, timing, and a timestamped output directory:

```bash
GO1_AIR_DEVICE=cpu \
GO1_AIR_DTYPE=float32 \
bash examples/offline_inference/go1_air/run.sh --collect
```

Results are written to `GO1_AIR_OUTPUT_ROOT/run_YYYYmmdd_HHMMSS/`.

## Output files

| File | Description |
|---|---|
| `actions.npy` | Raw action tensor (NumPy format) |
| `metadata.json` | dtype, shape, timing, seed, model path, commit hash |
| `env.json` | Python / torch / vllm / vllm-omni / transformers versions |
| `gpu.json` | nvidia-smi snapshot or `available: false` |
| `comparison.json` | (optional) upstream action comparison metrics |
| `forward_latency.json` | (optional) benchmark latency statistics |

## Optional upstream action comparison

Pass `--upstream-action-path` to `end2end.py` with a `.npy` or `.pt` file
containing an upstream action tensor.  The harness reports `max_abs_error`,
`mean_abs_error`, `mse`, and SHA-256 digests for both tensors.

## Running gated tests

```bash
# CPU stub test (always eligible, no GPU or checkpoint needed):
pytest -q tests/e2e/offline_inference/test_go1_air.py::test_go1_air_offline_cpu_stub

# Real-checkpoint test (requires H100/MI325 GPU + GO1_AIR_MODEL_DIR):
pytest -q tests/e2e/offline_inference/test_go1_air.py \
    -m advanced_model --run-level advanced_model
```

## Known limitations

* **GPU real-checkpoint inference** requires an NVIDIA driver >= 570 (CUDA 13.x
  API).  The GO-1-Air vllm-omni integration targets the vllm 0.20.x release
  which is built against CUDA 13.  CPU stub mode works on any platform.
* Tokenizer loading may use `trust_remote_code=True` when loading from the
  GO-1-Air checkpoint directory.  Review the model card before deployment.

## Notes

* The full open-loop evaluation harness (dataset loader, deterministic noise,
  result archiving) is added in a follow-up PR — see
  `examples/offline_inference/internvla_a1/` for the structure that will be
  mirrored.
* The GO-1-Air weights are licensed under **CC BY-NC-SA 4.0**; the
  vllm-omni integration code is Apache-2.0 and contains no upstream model
  code. Downstream commercial deployment of the weights is governed by
  AgiBot's license — see the model card before shipping.
