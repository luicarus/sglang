# QuantAssay adaptive Prefill experiment

This branch extends the SGLang 0.5.3 QuantAssay scheduler branch with optional
iteration-level chunk budgets and native KV observations. The new feature is
off by default. Existing LPM and LPM aging sorting remain unchanged.

For this experiment, explicitly select `--schedule-policy lpm` for every case.
Use the same model, mixed-chunk setting, attention backend, graph capture,
memory fraction, request cap, warmup and arrival trace in all cases.

## Controls

| Flag | Default | Meaning |
|---|---:|---|
| `--adaptive-prefill` | off | Select a gross chunk budget for each Prefill attempt |
| `--chunked-prefill-size` | native | Upper bound, or fixed budget when adaptive mode is off |
| `--adaptive-prefill-mid-tokens` | 256 | Budget with two active Decode requests |
| `--adaptive-prefill-min-tokens` | 128 | Budget with at least three active Decode requests |
| `--trace-prefill-budgets` | off | Emit JSON preparation and admission-stop events |

With zero or one active Decode request, adaptive mode uses the chunk ceiling.
The native PrefillAdder subtracts the mixed Decode reservation once. Adaptive
mode aligns the remaining Prefill budget down to pages and leaves at least one
page when the ceiling permits it. Native KV reservation and request-slot
constraints still decide admission; this is not an allocation or eviction
rewrite. Adaptive mode currently requires mixed chunks, PP=1, and ordinary
non-disaggregated serving. Budgets must fit the ceiling and page alignment.

## Evidence

`QUANTASSAY_PREFILL_EVENT` log records include the selected gross budget,
Decode reservation, active requests, logical and page-charged Prefill tokens,
prepared request IDs, mixed Decode count, and the actual native stop reason.
KV free and evictable tokens are separate, with full/SWA pools distinguished.
Unavailable observations are marked explicitly.

These events describe **host-side preparation**, not GPU execution duration.
Repeated identical blocked states are suppressed, so event counts cannot be
read as failed request counts or total rejected attempts. Actual eviction
counts are unavailable. Warmup events must be excluded from timed results.

QuantAssay replays phased traffic and joins source events, per-request queue
times, streamed chunk gaps and sampled metrics. Compare fixed 128, fixed 512,
and adaptive 128–512 before interpreting performance. A single run cannot
establish stable improvement; inspect actual budget variation and resource
warnings before deciding whether independent repeats are justified.
