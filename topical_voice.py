"""Topical voice DNA pass (Phase 1b.4).

Extracts what subjects the creator covers, what positions they hold, what
frameworks they invoke, and who they reference. Thin wrapper around
dna.run_extraction_pass holding pass-specific prompts.
"""

from __future__ import annotations

from typing import Any

from supabase import Client

from dna import LlmCall, run_extraction_pass


PASS_NAME = "topical_voice"


TOPICAL_VOICE_SYSTEM_PROMPT = """You are an expert in creator voice analysis. You will be given a set of YouTube video transcripts from a single creator. Your task is to identify topical voice patterns — what subjects the creator covers, what positions they hold, what frameworks they invoke, and who they reference.

Extract claims about these four categories of topical voice:

1. **Coverage** — DOMAINS the creator covers across their catalog. Coverage claims must include either (a) relative weighting across multiple domains, (b) the lens through which the creator approaches their domain, or (c) what makes their coverage distinctive. Single-word topics like "covers productivity" are NOT acceptable — they're reductive boilerplate. Force yourself to write a coverage claim that another creator in the same broad space could not also claim.

2. **Stances** — falsifiable POSITIONS the creator holds within their domains. If you can phrase it as "X argues Y" or "X champions A over B", it's a stance.

3. **Frameworks** — NAMED systems, concepts, or methodologies the creator repeatedly invokes (e.g., "active recall", "second brain", "deep work"). A framework is a thing-in-itself the creator returns to, distinct from a position.

4. **Influences** — NAMED people, books, or sources the creator references repeatedly (e.g., "James Clear", "Atomic Habits", "Cal Newport").

BOUNDARY HEURISTIC:
- Names a verbatim concept (e.g., "active recall") → Framework
- Asserts a position about the world → Stance
- Names a domain → Coverage
- Names a person or book → Influence

CITATION SEMANTICS: every claim must include at least one piece of evidence with a verbatim exact_quote from the transcript. The cited quote demonstrates ONE INSTANCE of the pattern, not proof that the pattern recurs. confidence_score should reflect HOW WIDESPREAD the pattern is across the bucket's transcripts, NOT the quality of the single citation.

GOOD claims are SPECIFIC and CITABLE:
Coverage:
- "Primarily covers personal productivity with a strong educational/student-success lens; finance and tech are secondary domains"
- "Spans three core domains: study/learning techniques, productivity systems, and lifestyle/finance content"

Stances:
- "Argues active recall is the single most effective study technique"
- "Advocates habit-based change over goal-setting, recurring framing in Atomic Habits review and adjacent content"
- "Prefers Kindle over physical books for fiction; recommends physical for reference/non-fiction"

Frameworks:
- "Repeatedly invokes 'active recall' as a personally championed study concept"
- "Uses 'second brain' as a recurring productivity framework, often paired with PKM"

Influences:
- "Frequently cites James Clear's 'Atomic Habits' as a touchstone for habit-formation arguments"
- "References Cal Newport repeatedly in deep-work content; Newport functions as a credibility anchor"

BAD claims are GENERIC, VAGUE, or REDUCTIVE:
- "Covers productivity" — single-word, reductive (Coverage anti-boilerplate)
- "Has many topics" — vague
- "Has opinions about studying" — vague (Stance)
- "Talks about productivity frameworks" — vague (Framework)
- "Reads books" / "Likes Atomic Habits" — too thin (Influence)

For each claim:
- claim_text: a one-sentence description of the topical pattern.
- evidence: at least one {video_id, exact_quote}. exact_quote MUST be a verbatim substring from the transcript exemplifying the pattern. video_id MUST be one of the IDs provided in the user message's <video id="..."> tags.
- confidence_score: 0.0 to 1.0, reflecting how widespread the pattern is across the bucket's transcripts.

Use the record_claims tool to return your findings. Aim for breadth — extract every distinctive topical pattern you can find. Synthesis across buckets will deduplicate later."""


TOPICAL_VOICE_SYNTHESIS_PROMPT = """You are merging topical voice claims that were extracted independently from multiple buckets of transcripts from the same creator. Many patterns will appear in multiple buckets phrased differently — especially Coverage claims (one bucket might say "personal productivity", another "personal effectiveness" for the same domain) and Influences ("James Clear" and "Atomic Habits author" are the same influence). Your job is to deduplicate them into canonical claims.

You will see input claims as a flat numbered list. For each canonical claim you produce, return:

- claim_text: the canonical phrasing of the pattern. Pick the clearest of the input phrasings, or write a better one.
- confidence_score: 0.0 to 1.0, reflecting how widespread the pattern is across all the input claims. A pattern appearing in many input claims should have higher confidence than one appearing only once.
- source_claim_indexes: a list of integer indexes into the input list that this canonical claim merges from. At least one index required.

Rules:
- Two claims describe the same pattern if they identify the same domain, position, framework, or influence — even when the phrasing or specific terms differ. BE PERMISSIVE in merging, especially for Coverage (different bucket-level phrasings of the same domain) and Influences (the same person referenced under different aliases). Merging too aggressively is less harmful than splitting hair-distinctions.
- Every canonical claim must have at least one source_claim_index. Do NOT invent canonical claims with no source.
- An input claim should typically map to exactly one canonical claim, or be dropped if it's too vague to keep.
- You do NOT emit evidence — the orchestrator unions evidence from the source claims you reference. Just choose which source claims to merge.

Use the record_canonical_claims tool."""


def run_pass(
    creator_id: str,
    sb: Client,
    *,
    bucket_size: int = 10,
    llm_call: LlmCall | None = None,
) -> dict[str, Any]:
    """Run the topical voice extraction pass end-to-end.

    COST: ~$1.20-$1.40 per invocation against a typical channel. Real
    Anthropic API calls (claude-sonnet-4-6) — do NOT invoke casually
    expecting $0. For zero-cost testing, pass a stub `llm_call`.
    """
    return run_extraction_pass(
        creator_id,
        sb,
        pass_name=PASS_NAME,
        system_prompt=TOPICAL_VOICE_SYSTEM_PROMPT,
        synthesis_prompt=TOPICAL_VOICE_SYNTHESIS_PROMPT,
        bucket_size=bucket_size,
        llm_call=llm_call,
    )
