# QuantAssay scheduler branch

This branch maintains QuantAssay's waiting-aware LPM scheduler on top of
SGLang v0.5.3, base commit `a4a3d8239365d2fc16e2dc707605373ea3f35ded`.
It is a research branch in the existing SGLang fork. Contribution branches
and the fork's default branch are managed independently.

## Runtime behavior

The default scheduler is `lpm-aging`. Requests below a queue-wait threshold
retain longest-prefix-match ordering and the existing in-batch duplicate
deprioritization. Requests at or above the threshold are prioritized in
queue-entry order. The threshold defaults to 1000 ms and is configured by
`SGLANG_LPM_MAX_WAIT_MS`. This is a priority threshold, not a hard latency bound.
The existing FCFS fallback for large queues is retained. Other upstream policy
options remain available for API compatibility and controlled evaluation.

The modification changes CPU-side queue ordering. KV allocation, model
kernels, request admission constraints, and serving APIs retain their upstream
implementation. The previous prototype's quota extension is not implemented.

## Run directly

Use a compatible SGLang 0.5.3 dependency environment. Select this branch's
Python sources without reinstalling the serving dependencies:

```bash
export PYTHONPATH="<fork>/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_LPM_MAX_WAIT_MS=1000
python -m sglang.launch_server --model-path <model-snapshot> \
  --attention-backend triton --dtype bfloat16 \
  --context-length 512 --max-running-requests 4 \
  --mem-fraction-static 0.8 --cuda-graph-max-bs 4
```

On toolchains where FlashInfer JIT is unavailable, set
`SGLANG_IS_FLASHINFER_AVAILABLE=false`. Parameters above describe the small-model
evaluation setup; other models/hardware require their own measurements.

## QuantAssay integration

[QuantAssay](https://github.com/luicarus/QuantAssay) owns workload construction,
native scheduler observation and experiment evidence. Its `--engine-source`
argument points to this checkout's `python/` directory. Use fixed commits and
new run directories for reproducibility. The controller records the engine's
Git revision, dirty state and actual runtime source hashes.

The prototype's synthetic warm-cache comparisons showed reduced waiting for
independent prefixes and increased waiting for shared prefixes. Throughput gains
were inconsistent; the supplementary cold-cache comparison did not reduce the
maximum overall wait. This branch does not claim general acceleration,
starvation freedom, production safety or upstream acceptance. Earlier evidence
was collected from an isolated installed-source prototype; this Git port has
not undergone a new GPU performance campaign.

Keep upstream PR fixes on separate branches based on the appropriate upstream
revision. Do not submit this entire research branch as an upstream PR.

SGLang's source attribution and Apache-2.0 license are retained.
