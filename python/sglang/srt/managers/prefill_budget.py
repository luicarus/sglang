# Copyright 2026 QuantAssay contributors
# Licensed under the Apache License, Version 2.0.
"""Decode-pressure chunk budgets and read-only native KV observations."""

import json
import logging
import time

from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator

logger = logging.getLogger(__name__)
EVENT_MARKER = "QUANTASSAY_PREFILL_EVENT "


def kv_snapshot(scheduler):
    allocator, cache = scheduler.token_to_kv_pool_allocator, scheduler.tree_cache
    try:
        if isinstance(allocator, SWATokenToKVPoolAllocator):
            pools = {
                "full": {"free_tokens": int(allocator.full_available_size()),
                         "evictable_tokens": int(cache.full_evictable_size())},
                "swa": {"free_tokens": int(allocator.swa_available_size()),
                        "evictable_tokens": int(cache.swa_evictable_size())},
            }
        else:
            pools = {"full": {"free_tokens": int(allocator.available_size()),
                              "evictable_tokens": int(cache.evictable_size())}}
        return {"status": "ok", "unit": "tokens", "pools": pools}
    except (AttributeError, NotImplementedError, TypeError, ValueError) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}


def budget_context(scheduler, running_bs):
    args = scheduler.server_args
    active = sum(not req.finished() for req in scheduler.running_batch.reqs)
    reserved = running_bs if scheduler.is_mixed_chunk else 0
    ceiling = scheduler.chunked_prefill_size
    budget = ceiling
    adaptive = args.adaptive_prefill
    if adaptive:
        proposed = (ceiling if active <= 1 else args.adaptive_prefill_mid_tokens
                    if active == 2 else args.adaptive_prefill_min_tokens)
        # Native PrefillAdder subtracts decode reservation exactly once. Leave
        # at least one aligned page when the ceiling permits it.
        budget = min(ceiling, max(proposed, reserved + scheduler.page_size))
        net = max(0, budget - reserved) // scheduler.page_size * scheduler.page_size
        budget = reserved + net
    scheduler._qa_prefill_attempt = getattr(scheduler, "_qa_prefill_attempt", 0) + 1
    return {
        "attempt_id": scheduler._qa_prefill_attempt,
        "adaptive": adaptive, "chunk_ceiling_tokens": ceiling,
        "gross_chunk_budget_tokens": budget,
        "decode_reserved_tokens": reserved,
        "net_chunk_budget_tokens": None if budget is None else budget - reserved,
        "active_decode_reqs": active, "running_batch_size": running_bs,
        "waiting_reqs": len(scheduler.waiting_queue), "page_size": scheduler.page_size,
        "candidate_request_ids": [req.rid for req in scheduler.waiting_queue[:8]],
        "chunk_request_id": scheduler.chunked_req.rid if scheduler.chunked_req else None,
        "kv_before": kv_snapshot(scheduler) if args.trace_prefill_budgets else None,
    }


def emit_event(scheduler, context, status, *, reason=None, adder=None,
               request_ids=None, logical_tokens=None, mixed_decode_reqs=0):
    if not scheduler.server_args.trace_prefill_budgets:
        return
    if status == "blocked":
        # Suppress repeated decode-loop attempts with an unchanged block state.
        signature = (reason, context["running_batch_size"], context["waiting_reqs"],
                     context["chunk_request_id"], tuple(context["candidate_request_ids"]),
                     context["gross_chunk_budget_tokens"])
        if signature == getattr(scheduler, "_qa_last_prefill_block", None):
            return
        scheduler._qa_last_prefill_block = signature
    else:
        scheduler._qa_last_prefill_block = None
    admission_budget = context.get("admission_kv_budget")
    if admission_budget is None and adder is not None:
        admission_budget = {"reserved_remaining_tokens": adder.rem_total_tokens,
                            "current_remaining_tokens": adder.cur_rem_tokens,
                            "sample_moment": "admission_stop_before_native_batch_allocation"}
    event = {
        **context, "status": status, "stop_reason": reason,
        "request_ids": request_ids or [], "logical_prefill_tokens": logical_tokens,
        "mixed_decode_reqs": mixed_decode_reqs,
        "charged_prefill_tokens": adder.log_input_tokens if adder is not None else None,
        "admission_kv_budget": admission_budget,
        "kv_after": kv_snapshot(scheduler),
        "server_monotonic_seconds": time.perf_counter(),
        "clock_id": time.get_clock_info("perf_counter").implementation,
        "scope": "host-side prefill preparation, not GPU execution timing",
    }
    logger.info(EVENT_MARKER + "%s", json.dumps(event, sort_keys=True, allow_nan=False))
