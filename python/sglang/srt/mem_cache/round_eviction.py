"""Bounded, queue-visible marginal Prefill-round eviction for full RadixCache.

Scores estimate request chunks, not GPU iterations or measured execution time.
Matching is read-only: never refresh LRU, split nodes, or materialize CUDA indices.
"""

import heapq
import json
import logging
import time

logger = logging.getLogger(__name__)


def queued_demands(scheduler):
    args = scheduler.server_args
    reserved = len(scheduler.running_batch.reqs) if scheduler.is_mixed_chunk else 0
    budget = (scheduler.chunked_prefill_size - reserved) // scheduler.page_size * scheduler.page_size
    if budget <= 0:
        return []
    return [{"rid": r.rid, "tokens": list(r.origin_input_ids) + list(r.output_ids),
             "extra_key": r.extra_key, "budget": budget}
            for r in scheduler.waiting_queue if not r.finished()][:args.kv_eviction_lookahead]


def match_demands(cache, demands):
    from sglang.srt.mem_cache.radix_cache import RadixKey
    result = []
    for request in demands:
        tokens = request["tokens"]
        limit = max(0, len(tokens) - 1) // cache.page_size * cache.page_size
        key = RadixKey(tokens[:limit], request["extra_key"])
        node, depth, starts = cache.root_node, 0, {}
        while len(key):
            child = node.children.get(cache.get_child_key_fn(key))
            if child is None or child.value is None:
                break
            matched = cache.key_match_fn(child.key, key)
            if not matched:
                break
            starts[child] = depth
            depth += matched
            if matched < len(child.key):
                break
            key, node = key[matched:], child
        result.append({"rid": request["rid"], "length": len(tokens),
                       "budget": request["budget"], "prefix": depth, "starts": starts})
    return result


def marginal(node, demands):
    rounds, tokens = 0, 0
    for req in demands:
        before = req["prefix"]
        after = min(before, req["starts"].get(node, before))
        b = req["budget"]
        rounds += ((max(1, req["length"] - after) + b - 1) // b
                   - (max(1, req["length"] - before) + b - 1) // b)
        tokens += before - after
    return rounds, tokens


def evict(cache, requested):
    started = time.perf_counter()
    free_before = cache.token_to_kv_pool_allocator.available_size()
    demands = match_demands(cache, cache.round_demand_provider())
    heap = [(cache.eviction_strategy.get_priority(n), n) for n in cache._collect_leaves()]
    heapq.heapify(heap)
    freed, decisions = 0, []
    while freed < requested and heap:
        candidates = []
        while heap and len(candidates) < cache.round_candidate_limit:
            entry = heapq.heappop(heap)
            node = entry[1]
            if node is not cache.root_node and node.lock_ref == 0 and node.value is not None:
                candidates.append(entry)
        if not candidates:
            break
        costs = [marginal(n, demands) for _, n in candidates]
        index = 0
        if cache.round_eviction_mode == "prefill-rounds":
            index = min(range(len(candidates)), key=lambda i: (
                costs[i][0] / len(candidates[i][1].value),
                costs[i][1] / len(candidates[i][1].value),
                candidates[i][0]))
        _, node = candidates.pop(index)
        for entry in candidates:
            heapq.heappush(heap, entry)
        count = len(node.value)
        decisions.append({"node_id": node.id, "freed_tokens": count,
                          "estimated_extra_rounds": costs[index][0],
                          "extra_uncached_tokens": costs[index][1],
                          "lru_extra_rounds": costs[0][0],
                          "lru_extra_uncached_tokens": costs[0][1],
                          "different_from_lru": index != 0})
        # Update the snapshot after EVERY victim, including newly exposed parents.
        for req in demands:
            req["prefix"] = min(req["prefix"], req["starts"].get(node, req["prefix"]))
        cache.token_to_kv_pool_allocator.free(node.value)
        freed += count
        cache._delete_leaf(node)
        if not node.parent.children:
            heapq.heappush(heap, (cache.eviction_strategy.get_priority(node.parent), node.parent))
        cache._record_remove_event(node)
    elapsed = (time.perf_counter() - started) * 1000
    if cache.trace_round_eviction:
        logger.info("QUANTASSAY_KV_EVICTION %s", json.dumps({
            "mode": cache.round_eviction_mode, "requested_tokens": requested,
            "actual_freed_tokens": freed, "free_before": free_before,
            "free_after": cache.token_to_kv_pool_allocator.available_size(),
            "queue_request_ids": [r["rid"] for r in demands],
            "net_budgets": [r["budget"] for r in demands], "decisions": decisions,
            "host_eviction_ms": elapsed, "server_monotonic_seconds": time.perf_counter(),
            "clock_id": time.get_clock_info("perf_counter").implementation,
            "scope": "queue-snapshot request-chunk proxy; not GPU time or global batch count",
        }, allow_nan=False, sort_keys=True))
