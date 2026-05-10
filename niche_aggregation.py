"""Cross-creator pattern aggregation (Phase 1c.5).

For one managed creator: pull the LATEST complete peer_pattern_extractions
for each peer in the pool, ask Sonnet 4.6 to identify patterns appearing
in ≥2 distinct peers, persist the structured cross-peer aggregation to
niche_pattern_aggregations.

Index-based LLM I/O with server-side resolution (mirrors dna.py
synthesize_claims): the model emits source_indexes + evidence index
references, this orchestrator resolves them to full peer-occurrence
objects and verbatim evidence strings. Prevents the model from
hallucinating peer_creator_ids or evidence quotes.

Hardcoded prompt mirrors the DNA-pass + 1c.4 pattern-extraction
convention — analysis prompts in Python, content-generation prompts in
prompt_templates.
"""

from __future__ import annotations

import os
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from anthropic import Anthropic
from supabase import Client


MODEL = "claude-sonnet-4-6"

PRICE_INPUT_PER_TOKEN = 3.0 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 15.0 / 1_000_000
PRICE_CACHE_WRITE_PER_TOKEN = 3.75 / 1_000_000
PRICE_CACHE_READ_PER_TOKEN = 0.3 / 1_000_000

MIN_PEERS_REQUIRED = 2

CATEGORIES = ("title", "hook", "structural", "topical")


SYSTEM_PROMPT = """You are an expert content analyst at a creator-management agency. Your job: synthesize cross-peer patterns from a set of per-peer pattern extractions in the managed creator's niche.

You will see:
- The MANAGED CREATOR's identity (for relevance framing).
- 2-N PEERS, each with their per-category extracted patterns (title, hook, structural, topical). Each pattern carries a description, evidence items, the peer's confidence, and the peer's reasoning.

Your job: identify patterns that appear in 2+ DISTINCT peers and synthesize them into cross-creator patterns that represent niche-level signal — the patterns the agency should treat as "what's working in this niche right now."

For each category (title, hook, structural, topical) independently:
1. Group patterns from different peers that describe the SAME or near-same phenomenon. Loose semantic matches don't count — "active recall" (Peer A) and "spaced practice" (Peer B) are different study techniques, not the same pattern. "Numbered list titles with retention metrics" (Peer A) and "Numbered list with study-time figures" (Peer B) ARE the same pattern.
2. For each group, write ONE synthesized pattern_description that captures the shared phenomenon — not a copy of any single peer's wording.
3. Reference the source patterns by their input index — these become peer_occurrences in the output. Use source_indexes for the list of input patterns this cross-peer pattern aggregates.
4. Pick 3-5 best evidence examples drawn FROM the source patterns' evidence lists — cite each as (source_index, evidence_index in that source). Do not invent evidence.
5. Calibrate aggregate_confidence based on:
   - cross_peer_count (more peers showing it = stronger signal; 4 peers > 2 peers)
   - peer-level confidences in the source patterns (high-confidence sources lift the aggregate; weak sources cap it)
   - evidence quality (concrete + verbatim = stronger; vague = weaker)
   Examples: 2 peers with marginal confidence (~0.6) → aggregate ≤ 0.7. 4 peers all at ~0.9 with strong evidence → aggregate up to 0.95.
6. Reasoning: 1-2 sentences explaining why this is real cross-peer signal vs. coincidence.

HARD REQUIREMENTS (drop the pattern if any fails):
- source_indexes MUST contain entries from ≥2 DISTINCT peer_creator_ids. A single peer's two patterns don't aggregate — that's still one peer.
- source_indexes MUST be valid indexes into the corresponding category's input list. Do not invent indexes.
- synthesized_evidence_indexes MUST reference valid (source_index, evidence_index) pairs from the actual input. Each evidence_index must be < the length of that source's evidence list.
- 3-5 synthesized_evidence_indexes per pattern.
- Aim for 5-15 cross-peer patterns per category. If a category has no real cross-peer signal, return an EMPTY array — do not invent matches to hit a count.

Edge cases:
- If peers are from very different niches, you may legitimately find few or zero cross-peer patterns. Return what's actually there. Empty arrays are valid output.
- If a pattern appears strongly in just ONE peer (high evidence, high confidence) — it is NOT a cross-peer pattern. Do not include it. The single-peer signal already lives in that peer's own extraction.
- A peer with empty input arrays in some category contributes nothing to that category — focus on peers that have patterns there.
- Diversity across categories: if the input shows signal in all four categories, your output should too. Don't return 15 title patterns and 0 of others if hook/structural/topical genuinely have cross-peer matches.

Use the record_niche_aggregation tool to return your output."""


EVIDENCE_REF_SCHEMA = {
    "type": "object",
    "properties": {
        "source_index": {"type": "integer", "minimum": 0},
        "evidence_index": {"type": "integer", "minimum": 0},
    },
    "required": ["source_index", "evidence_index"],
}

CROSS_PEER_PATTERN_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern_description": {"type": "string"},
        "source_indexes": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 2,
        },
        "synthesized_evidence_indexes": {
            "type": "array",
            "items": EVIDENCE_REF_SCHEMA,
            "minItems": 3,
            "maxItems": 5,
        },
        "aggregate_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
    "required": [
        "pattern_description",
        "source_indexes",
        "synthesized_evidence_indexes",
        "aggregate_confidence",
        "reasoning",
    ],
}

RECORD_NICHE_AGGREGATION_TOOL = {
    "name": "record_niche_aggregation",
    "description": (
        "Synthesize cross-peer patterns across four categories using "
        "index-based references into the input pattern lists."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title_patterns": {
                "type": "array",
                "items": CROSS_PEER_PATTERN_SCHEMA,
            },
            "hook_patterns": {
                "type": "array",
                "items": CROSS_PEER_PATTERN_SCHEMA,
            },
            "structural_patterns": {
                "type": "array",
                "items": CROSS_PEER_PATTERN_SCHEMA,
            },
            "topical_patterns": {
                "type": "array",
                "items": CROSS_PEER_PATTERN_SCHEMA,
            },
        },
        "required": [
            "title_patterns",
            "hook_patterns",
            "structural_patterns",
            "topical_patterns",
        ],
    },
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _xml_attr(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _fetch_managed_creator(sb: Client, creator_id: str) -> dict:
    """Creator info + first platform_account for handle / sub count."""
    creator = (
        sb.table("creators")
        .select("id, name")
        .eq("id", creator_id)
        .single()
        .execute()
    ).data
    if not creator:
        raise ValueError(f"creator not found: {creator_id}")

    pa = (
        sb.table("platform_accounts")
        .select(
            "platform_account_id, platform_handle, display_name, "
            "subscriber_count"
        )
        .eq("creator_id", creator_id)
        .limit(1)
        .execute()
    ).data
    pa_row = pa[0] if pa else {}

    return {
        "id": creator["id"],
        "name": creator["name"],
        "handle": pa_row.get("platform_handle"),
        "display_name": pa_row.get("display_name") or creator["name"],
        "subscriber_count": pa_row.get("subscriber_count"),
    }


def _fetch_latest_extractions(
    sb: Client, creator_id: str
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Returns (extractions_by_peer_id, peer_info_by_peer_id).
    extractions_by_peer_id: peer_creator_id -> latest complete extraction.
    peer_info_by_peer_id: peer_creator_id -> { handle, display_name }.
    """
    pool_rows = (
        sb.table("creator_peer_pool")
        .select("peer_creator_id")
        .eq("creator_id", creator_id)
        .execute()
    ).data or []
    peer_ids = [r["peer_creator_id"] for r in pool_rows]
    if not peer_ids:
        return {}, {}

    extractions_by_peer: dict[str, dict] = {}
    for peer_id in peer_ids:
        rows = (
            sb.table("peer_pattern_extractions")
            .select(
                "id, peer_creator_id, status, title_patterns, hook_patterns, "
                "structural_patterns, topical_patterns, created_at"
            )
            .eq("peer_creator_id", peer_id)
            .eq("status", "complete")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ).data or []
        if rows:
            extractions_by_peer[peer_id] = rows[0]

    peer_creators = (
        sb.table("peer_creators")
        .select("id, handle, display_name, platform_account_id")
        .in_("id", list(extractions_by_peer.keys()))
        .execute()
    ).data or []
    peer_info_by_id = {p["id"]: p for p in peer_creators}

    return extractions_by_peer, peer_info_by_id


def _build_category_lists(
    extractions_by_peer: dict[str, dict],
    peer_info_by_id: dict[str, dict],
) -> dict[str, list[dict]]:
    """Flatten per-peer per-category patterns into indexed lists. Each
    entry carries the peer's identity so the LLM can recognize who said
    what, and the orchestrator can resolve indexes to full peer
    occurrences.
    """
    lists: dict[str, list[dict]] = {cat: [] for cat in CATEGORIES}
    for peer_id, ext in extractions_by_peer.items():
        peer_info = peer_info_by_id.get(peer_id, {})
        for cat in CATEGORIES:
            patterns = ext.get(f"{cat}_patterns") or []
            for p in patterns:
                lists[cat].append(
                    {
                        "peer_creator_id": peer_id,
                        "peer_handle": peer_info.get("handle") or "",
                        "peer_display_name": peer_info.get("display_name")
                        or "",
                        "description": p.get("pattern") or "",
                        "evidence": list(p.get("evidence") or []),
                        "confidence": p.get("confidence") or 0,
                        "reasoning": p.get("reasoning") or "",
                    }
                )
    return lists


def _build_user_message(
    creator: dict,
    peer_info_by_id: dict[str, dict],
    category_lists: dict[str, list[dict]],
) -> str:
    parts: list[str] = []

    parts.append("<managed_creator>")
    parts.append(f"  <handle>{_xml_attr(creator.get('handle'))}</handle>")
    parts.append(
        f"  <display_name>{_xml_attr(creator.get('display_name'))}</display_name>"
    )
    parts.append(
        f"  <subscriber_count>{creator.get('subscriber_count') or 0}</subscriber_count>"
    )
    parts.append("</managed_creator>")
    parts.append("")

    parts.append("<peers>")
    for peer_id, peer in peer_info_by_id.items():
        parts.append(
            f'  <peer creator_id="{_xml_attr(peer_id)}" '
            f'handle="{_xml_attr(peer.get("handle") or "")}" '
            f'display_name="{_xml_attr(peer.get("display_name") or "")}" />'
        )
    parts.append("</peers>")
    parts.append("")

    for cat in CATEGORIES:
        parts.append(f"<{cat}_patterns>")
        for i, p in enumerate(category_lists[cat]):
            parts.append(
                f'  <pattern index="{i}" '
                f'peer_creator_id="{_xml_attr(p["peer_creator_id"])}" '
                f'peer_handle="{_xml_attr(p["peer_handle"])}" '
                f'peer_confidence="{p["confidence"]}">'
            )
            parts.append(
                f"    <description>{_xml_attr(p['description'])}</description>"
            )
            for ei, ev in enumerate(p["evidence"]):
                parts.append(
                    f'    <evidence index="{ei}">{_xml_attr(ev)}</evidence>'
                )
            if p["reasoning"]:
                parts.append(
                    f"    <reasoning>{_xml_attr(p['reasoning'])}</reasoning>"
                )
            parts.append("  </pattern>")
        parts.append(f"</{cat}_patterns>")
        parts.append("")

    return "\n".join(parts)


def _call_anthropic(user_message: str) -> dict:
    client = Anthropic()
    with client.messages.stream(
        model=MODEL,
        max_tokens=12000,  # ~60 cross-peer patterns × ~150 tokens each
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[RECORD_NICHE_AGGREGATION_TOOL],
        tool_choice={
            "type": "tool",
            "name": RECORD_NICHE_AGGREGATION_TOOL["name"],
        },
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Anthropic response hit max_tokens "
            f"(output={response.usage.output_tokens}); tool call truncated."
        )

    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and block.name == RECORD_NICHE_AGGREGATION_TOOL["name"]
        ):
            tool_input = (
                dict(block.input) if isinstance(block.input, dict) else {}
            )
            usage = response.usage
            tool_input["_anthropic"] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation_input_tokens": (
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                ),
                "cache_read_input_tokens": (
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                ),
                "stop_reason": response.stop_reason,
                "model": MODEL,
            }
            return tool_input

    raise RuntimeError(
        f"Anthropic response did not include "
        f"{RECORD_NICHE_AGGREGATION_TOOL['name']} tool use; "
        f"stop_reason={response.stop_reason}"
    )


def _resolve_patterns(
    cross_patterns: list[dict],
    source_list: list[dict],
    peers_included: int,
    creator_id: str,
    category: str,
) -> list[dict]:
    """Resolve LLM-emitted index references to full peer-occurrence
    objects + verbatim evidence strings. Drops patterns failing the
    distinct-peer guard or with invalid indexes.
    """
    resolved: list[dict] = []
    for p in cross_patterns or []:
        source_indexes = p.get("source_indexes") or []
        evidence_refs = p.get("synthesized_evidence_indexes") or []

        # Validate source indexes are in range.
        if any(
            not isinstance(idx, int) or idx < 0 or idx >= len(source_list)
            for idx in source_indexes
        ):
            print(
                f"[niche_agg] {creator_id}: dropping {category} pattern with "
                f"out-of-range source_indexes={source_indexes} "
                f"(source_list len={len(source_list)})",
                flush=True,
            )
            continue

        peer_occurrences = []
        peer_ids_seen: set[str] = set()
        for idx in source_indexes:
            src = source_list[idx]
            peer_occurrences.append(
                {
                    "peer_creator_id": src["peer_creator_id"],
                    "peer_handle": src["peer_handle"],
                    "peer_display_name": src["peer_display_name"],
                    "peer_pattern": src["description"],
                    "peer_evidence": src["evidence"],
                    "peer_confidence": src["confidence"],
                }
            )
            peer_ids_seen.add(src["peer_creator_id"])

        # Distinct-peer guard. The schema requires source_indexes have
        # ≥2 entries, but we additionally require ≥2 DISTINCT peers.
        if len(peer_ids_seen) < MIN_PEERS_REQUIRED:
            print(
                f"[niche_agg] {creator_id}: dropping {category} pattern: "
                f"only {len(peer_ids_seen)} distinct peers in "
                f"source_indexes={source_indexes}",
                flush=True,
            )
            continue

        # Resolve evidence references to verbatim strings.
        synthesized_evidence: list[str] = []
        for ref in evidence_refs:
            sidx = ref.get("source_index")
            eidx = ref.get("evidence_index")
            if (
                not isinstance(sidx, int)
                or not isinstance(eidx, int)
                or sidx < 0
                or sidx >= len(source_list)
            ):
                continue
            src_evidence = source_list[sidx]["evidence"]
            if eidx < 0 or eidx >= len(src_evidence):
                continue
            synthesized_evidence.append(src_evidence[eidx])

        if len(synthesized_evidence) < 2:
            print(
                f"[niche_agg] {creator_id}: dropping {category} pattern: "
                f"only {len(synthesized_evidence)} valid evidence refs out "
                f"of {len(evidence_refs)} requested",
                flush=True,
            )
            continue

        cross_peer_count = len(peer_ids_seen)
        if cross_peer_count == peers_included:
            consistency = "unanimous"
        elif cross_peer_count > peers_included / 2:
            consistency = "majority"
        else:
            consistency = "partial"

        resolved.append(
            {
                "pattern_description": p.get("pattern_description"),
                "peer_occurrences": peer_occurrences,
                "cross_peer_count": cross_peer_count,
                "cross_peer_consistency": consistency,
                "aggregate_confidence": p.get("aggregate_confidence"),
                "synthesized_evidence": synthesized_evidence,
                "reasoning": p.get("reasoning"),
            }
        )

    # Sort by aggregate_confidence desc, then cross_peer_count desc.
    resolved.sort(
        key=lambda r: (
            -(r.get("aggregate_confidence") or 0),
            -(r.get("cross_peer_count") or 0),
        )
    )
    return resolved


def run_niche_aggregation(sb: Client, creator_id: str) -> dict[str, Any]:
    """One-shot cross-creator pattern aggregation for a managed creator.
    Raises ValueError if fewer than MIN_PEERS_REQUIRED peers have complete
    extractions — caller pre-flights, but defense in depth.
    """
    generation_run_id = str(uuid.uuid4())
    print(
        f"[niche_agg] {creator_id}: starting, generation_run_id={generation_run_id}",
        flush=True,
    )

    creator = _fetch_managed_creator(sb, creator_id)
    extractions_by_peer, peer_info_by_id = _fetch_latest_extractions(
        sb, creator_id
    )
    peers_included = len(extractions_by_peer)
    print(
        f"[niche_agg] {creator_id}: peers_with_extractions={peers_included}",
        flush=True,
    )

    if peers_included < MIN_PEERS_REQUIRED:
        raise ValueError(
            f"need at least {MIN_PEERS_REQUIRED} peers with complete "
            f"extractions, got {peers_included}"
        )

    category_lists = _build_category_lists(
        extractions_by_peer, peer_info_by_id
    )
    print(
        f"[niche_agg] {creator_id}: data assembled — "
        f"peers={peers_included}, "
        f"total_patterns(t/h/s/p)="
        f"{len(category_lists['title'])}/"
        f"{len(category_lists['hook'])}/"
        f"{len(category_lists['structural'])}/"
        f"{len(category_lists['topical'])}",
        flush=True,
    )

    inserted = (
        sb.table("niche_pattern_aggregations")
        .insert(
            {
                "creator_id": creator_id,
                "generation_run_id": generation_run_id,
                "status": "aggregating",
                "peers_included": peers_included,
                "peer_extraction_ids": [
                    ext["id"] for ext in extractions_by_peer.values()
                ],
                "extraction_model": MODEL,
            }
        )
        .execute()
    ).data
    aggregation_id = inserted[0]["id"] if inserted else None
    if not aggregation_id:
        raise RuntimeError("Failed to insert niche_pattern_aggregations row")
    print(
        f"[niche_agg] {creator_id}: aggregation_id={aggregation_id}",
        flush=True,
    )

    try:
        user_message = _build_user_message(
            creator, peer_info_by_id, category_lists
        )
        print(
            f"[niche_agg] {creator_id}: user_message len={len(user_message)} chars",
            flush=True,
        )

        result = _call_anthropic(user_message)
        meta = result.pop("_anthropic")
        print(
            f"[niche_agg] {creator_id}: anthropic call complete: "
            f"input={meta['input_tokens']}, output={meta['output_tokens']}, "
            f"cache_create={meta['cache_creation_input_tokens']}, "
            f"cache_read={meta['cache_read_input_tokens']}, "
            f"stop_reason={meta['stop_reason']}",
            flush=True,
        )

        cost = (
            meta["input_tokens"] * PRICE_INPUT_PER_TOKEN
            + meta["output_tokens"] * PRICE_OUTPUT_PER_TOKEN
            + meta["cache_creation_input_tokens"] * PRICE_CACHE_WRITE_PER_TOKEN
            + meta["cache_read_input_tokens"] * PRICE_CACHE_READ_PER_TOKEN
        )

        resolved_by_cat: dict[str, list[dict]] = {}
        for cat in CATEGORIES:
            resolved_by_cat[cat] = _resolve_patterns(
                result.get(f"{cat}_patterns") or [],
                category_lists[cat],
                peers_included,
                creator_id,
                cat,
            )
        print(
            f"[niche_agg] {creator_id}: parsed (post-resolve): "
            f"t={len(resolved_by_cat['title'])}, "
            f"h={len(resolved_by_cat['hook'])}, "
            f"s={len(resolved_by_cat['structural'])}, "
            f"p={len(resolved_by_cat['topical'])}",
            flush=True,
        )

        sb.table("niche_pattern_aggregations").update(
            {
                "status": "complete",
                "title_patterns": resolved_by_cat["title"],
                "hook_patterns": resolved_by_cat["hook"],
                "structural_patterns": resolved_by_cat["structural"],
                "topical_patterns": resolved_by_cat["topical"],
                "input_tokens": meta["input_tokens"],
                "output_tokens": meta["output_tokens"],
                "cache_creation_tokens": meta["cache_creation_input_tokens"],
                "cache_read_tokens": meta["cache_read_input_tokens"],
                "stop_reason": meta["stop_reason"],
                "cost_usd": round(cost, 4),
                "completed_at": _utcnow_iso(),
                "error": None,
            }
        ).eq("id", aggregation_id).execute()

        print(
            f"[niche_agg] {creator_id}: complete: "
            f"${cost:.4f}, peers_included={peers_included}",
            flush=True,
        )

        return {
            "ok": True,
            "aggregation_id": aggregation_id,
            "generation_run_id": generation_run_id,
            "cost_usd": round(cost, 4),
            "peers_included": peers_included,
            "title_patterns": len(resolved_by_cat["title"]),
            "hook_patterns": len(resolved_by_cat["hook"]),
            "structural_patterns": len(resolved_by_cat["structural"]),
            "topical_patterns": len(resolved_by_cat["topical"]),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(
            f"[niche_agg] {creator_id}: FAILED: "
            f"{type(e).__name__}: {e}\n{tb}",
            flush=True,
        )
        sb.table("niche_pattern_aggregations").update(
            {
                "status": "failed",
                "error": str(e)[:500],
                "completed_at": _utcnow_iso(),
            }
        ).eq("id", aggregation_id).execute()
        raise
