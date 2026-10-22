# Queue-visible marginal Prefill-round eviction

Experimental, opt-in policy on the QuantAssay SGLang 0.5.3 branch.
The new `--kv-eviction-mode prefill-rounds` chooses reclaimable prefix-cache
leaves according to the extra request chunks their removal would cause.
The default `native` keeps the original path when tracing is also disabled.

For a queued request of length L, retained prefix P and positive net chunk
budget B, the proxy is ceil(max(1, L-P)/B). For each victim, sum the marginal
increase across the visible queued requests and divide by freed tokens.
Tie-break with extra uncached tokens per freed token, then the original LRU
ordering. After each victim, shorten affected snapshot prefixes and rescore
the next candidate set, including newly exposed parents.

Controls:

| Flag | Default | Meaning |
|---|---:|---|
| `--kv-eviction-mode` | native | native or prefill-rounds |
| `--kv-eviction-candidates` | 32 | Oldest eligible leaves scored per selection |
| `--kv-eviction-lookahead` | 16 | First unfinished waiting requests considered |
| `--trace-kv-evictions` | off | Raw victim, allocation and host timing events |

The first implementation supports Python full-attention RadixCache, LRU,
fixed chunk budgets, non-speculative PP=1 non-disaggregated serving.
Queue matching is read-only: no timestamp refresh, tree split or CUDA-index
materialization. Native lock rules, whole-leaf freeing, parent exposure,
allocation checks, eviction target amounts and scheduling priority remain.
No queue demand or no positive net budget gives LRU behavior. Active Attention
KV is not pruned; this is reusable-prefix retention, not CPU offload.

Candidate scoring is bounded, but collecting leaves/building the heap still
visits the tree as in the original implementation. Round counts are request
chunk estimates, not global GPU iteration counts or measured time. Multiple
waiting requests can share a future recomputation, so the summed proxy can
overestimate actual additional work. A local LRU alternative in a log is
not a measured counterfactual run.

`QUANTASSAY_KV_EVICTION` records requested and actually freed token indices,
allocator free counts before/after, candidate decisions, snapshot budgets,
queued request IDs and host eviction time. Token counts do not mean physical
GPU memory was returned to the driver. Traced native runs also compute costs
for observation: their CPU difference does not measure the full overhead
relative to the completely uninstrumented original implementation.

## Initial evidence, 2026-10-22

Qwen3-0.6B BF16 / SGLang 0.5.3 / 4 GB RTX 3050 Ti Laptop / WSL2.
80 requests, eight repeated-prefix families, inputs around 416–512 tokens,
64 fixed output tokens, 8 requests/s, running cap 2, chunk 128, mixed chunk
off. The KV pool is explicitly capped at **2048 tokens**, not the device's
natural maximum capacity. Each pair replays the same trace with identical
source/model/controller identities and other actual serving arguments.

| Schedule / order | Cached tokens native → policy | Actual request chunks | Output tok/s |
|---|---:|---:|---:|
| LPM, native first | 31860 → 31860 | 104 → 104 | 214.118 → 214.789 |
| FCFS, native first | 5859 → 12663 | 284 → 228 | 179.080 → 184.844 |
| FCFS, policy first | 5859 → 12382 | 284 → 231 | 176.279 → 180.699 |

All six runs succeeded on 80/80 requests and generated 5120 output tokens.
Freed-index counts matched allocator free-count deltas in every timed event.
The LPM pair made no different victim choices: no additional benefit was
demonstrated there. The two FCFS pairs changed 259 and 257 victim choices,
reduced actual request chunks by 19.7% and 18.7%, and improved throughput by
3.2% and 2.5%. These are limited synthetic, capacity-controlled observations,
not a general speedup claim. Mode remains opt-in.
Tail latency did not improve consistently: the second policy FCFS run retained
a maximum ITL of 558.885 ms. No slow samples were removed.

Run IDs: `kv-rounds-{native,policy}-20261022-{a,b,c}` in QuantAssay evidence.
Measured engine source: `08b19e0`; subsequent documentation does not change
the runtime. No unit tests were added or run for this experiment.

## Design provenance

This branch implements its own marginal request-chunk selection rule.
The upstream policy code inspected at `afd0d7b47b40448341036cea37294fcf3cb2aabd`
already includes LRU/LFU/SLRU/Priority/T-LRU. Queue-visible next-use information
also appears in UniCache OPT-Online. The present prototype uses marginal
chunk-boundary cost rather than importing those policies. The review did not
find this exact rule; it does not establish worldwide novelty.
