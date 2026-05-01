"""Lexical voice DNA pass (Phase 1b.2).

Extracts the specific words, phrases, and verbal habits that make a
creator's voice recognizable. Thin wrapper around dna.run_extraction_pass
holding pass-specific prompts.
"""

from __future__ import annotations

from typing import Any

from supabase import Client

from dna import LlmCall, run_extraction_pass


PASS_NAME = "lexical_voice"


LEXICAL_VOICE_SYSTEM_PROMPT = """You are an expert in creator voice analysis. You will be given a set of YouTube video transcripts from a single creator. Your task is to identify lexical voice patterns — the specific words, phrases, and verbal habits that make this creator's voice recognizable.

Extract claims about these four categories of lexical voice:

1. **Recurring phrases** — transitions, signature openers and closers, repeated framings (e.g., "alright so", "the way I think about it is...", "let's get into it").
2. **Distinctive vocabulary** — idioms or specific word choices that feel personal vs. generic (e.g., "absolute banger", "fundamentally", domain-specific jargon used in a personal way).
3. **Audience address** — how the creator refers to viewers ("you guys", "team", "friends", "everyone"), or whether they mostly speak without addressing the audience at all.
4. **Verbal tics** — filler patterns and sentence-starters that recur regardless of topic (e.g., "right, so...", "the thing is...", "look —", trailing "you know").

GOOD claims are SPECIFIC and CITABLE:
- "Opens videos with the phrase 'Hey friends'"
- "Uses 'fundamentally' to introduce contested or load-bearing points"
- "Refers to viewers as 'you guys' rather than 'you' or 'everyone'"

BAD claims are GENERIC, VAGUE, or NON-LEXICAL:
- "Speaks clearly" — not lexical
- "Engages viewers" — not specific, not citable
- "Uses casual language" — vague, unfalsifiable

For each claim:
- claim_text: a one-sentence description of the pattern.
- evidence: at least one {video_id, exact_quote}. exact_quote MUST be a verbatim substring from the transcript demonstrating the pattern. video_id MUST be one of the IDs provided in the user message's <video id="..."> tags.
- confidence_score: 0.0 to 1.0, reflecting how widespread the pattern is across the transcripts in this bucket. A pattern appearing in most transcripts gets higher confidence than one appearing in a single transcript.

Use the record_claims tool to return your findings. Aim for breadth — extract every distinctive lexical pattern you can find, not just the strongest. Synthesis across buckets will deduplicate later."""


LEXICAL_VOICE_SYNTHESIS_PROMPT = """You are merging lexical voice claims that were extracted independently from multiple buckets of transcripts from the same creator. Many patterns will appear in multiple buckets phrased slightly differently. Your job is to deduplicate them into canonical claims.

You will see input claims as a flat numbered list. For each canonical claim you produce, return:

- claim_text: the canonical phrasing of the pattern. Pick the clearest of the input phrasings, or write a better one.
- confidence_score: 0.0 to 1.0, reflecting how widespread the pattern is across all the input claims. A pattern that appears in many input claims should have higher confidence than one appearing only once.
- source_claim_indexes: a list of integer indexes into the input list that this canonical claim merges from. At least one index required.

Rules:
- Two claims describe the same pattern if they identify the same lexical phenomenon, even if phrased differently. Merge them.
- Every canonical claim must have at least one source_claim_index. Do NOT invent canonical claims with no source.
- An input claim should typically map to exactly one canonical claim, or be dropped if it's too vague to keep.
- Be conservative: prefer merging similar claims over splitting them into hair-distinctions.
- You do NOT emit evidence — the orchestrator unions evidence from the source claims you reference. Just choose which source claims to merge.

Use the record_canonical_claims tool."""


def run_pass(
    creator_id: str,
    sb: Client,
    *,
    bucket_size: int = 10,
    llm_call: LlmCall | None = None,
) -> dict[str, Any]:
    """Run the lexical voice extraction pass end-to-end.

    COST: ~$1.40 per invocation against a typical ~95-transcript channel.
    Real Anthropic API calls (claude-sonnet-4-6) — do NOT invoke casually
    expecting $0. For zero-cost testing, pass a stub `llm_call`.
    """
    return run_extraction_pass(
        creator_id,
        sb,
        pass_name=PASS_NAME,
        system_prompt=LEXICAL_VOICE_SYSTEM_PROMPT,
        synthesis_prompt=LEXICAL_VOICE_SYNTHESIS_PROMPT,
        bucket_size=bucket_size,
        llm_call=llm_call,
    )
