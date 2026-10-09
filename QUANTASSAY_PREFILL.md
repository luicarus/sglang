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
Admission reservation remainders are captured before native batch allocation;
allocator observations after preparation are recorded separately.

These events describe **host-side preparation**, not GPU execution duration.
Repeated identical blocked states are suppressed, so event counts cannot be
read as failed request counts or total rejected attempts. Actual eviction
counts are unavailable. Warmup events must be excluded from timed results.

QuantAssay replays phased traffic and joins source events, per-request queue
times, streamed chunk gaps and sampled metrics. Compare fixed 128, fixed 512,
and adaptive 128–512 before interpreting performance. A single run cannot
establish stable improvement; inspect actual budget variation and resource
warnings before deciding whether independent repeats are justified.

## Initial service observations (2026-10-09)

Qwen3-0.6B BF16, RTX 3050 Ti Laptop 4 GB, SGLang 0.5.3, LPM, Triton attention,
mixed chunks, context 1024, running cap 4, graph cap 4, static memory 0.8.
Each case replayed the same 60 unique-prefix requests with 1/12/1 requests/s
arrival phases. Output lengths were held at 64/128/192 with ignore_eos, for
7680 tokens per case. Eight concurrent warmup requests generated 192 tokens
each, followed by a cache flush; the server seed was 42.

| Metric | Fixed 128 | Fixed 512 | Adaptive |
|---|---:|---:|---:|
| Low-phase long-input TTFT p50, ms | 125.991 | 55.411 | 53.826 |
| Overall TTFT p95, ms | 5809.734 | 4994.204 | 5832.543 |
| Overall ITL p95, ms | 10.243 | 9.839 | 10.248 |
| Output tokens/s | 182.150 | 182.136 | 182.176 |

Adaptive budgets actually varied: 33 prepared batches at 512, 6 at 256,
and 69 at 128. All cases succeeded on 60/60 requests and had matching
source/model/controller/traffic identities. Adaptive mode improved the
low-load long-input TTFT relative to fixed 128, but did not improve the
overall tail metrics relative to fixed 512. Throughput is arrival-limited
and nearly identical. The feature remains off by default.

There is only one valid run per case under this fixed-output protocol.
An earlier EOS-allowed series has a different protocol. One fixed-128 run
had an execution slowdown and 26 timeouts; it was retained for diagnosis
and rerun with unchanged code/configuration. Real early gaps of roughly
half a second remain in the adaptive and fixed-128 valid runs; their cause
is unresolved. These observations do not establish a stable improvement.
Observed admission stops were request-slot limits, rather than KV budget
exhaustion. KV admission, eviction and cancellation cleanup are unchanged.
