"""Structural voice DNA pass (Phase 1b.3).

Extracts how a creator organizes a video — opening, closing, transitions,
story insertion, and CTA patterns. Thin wrapper around
dna.run_extraction_pass holding pass-specific prompts.
"""

from __future__ import annotations

from typing import Any

from supabase import Client

from dna import LlmCall, run_extraction_pass


PASS_NAME = "structural_voice"


STRUCTURAL_VOICE_SYSTEM_PROMPT = """You are an expert in creator voice analysis. You will be given a set of YouTube video transcripts from a single creator. Your task is to identify structural voice patterns — how the creator organizes a video: opens, closes, transitions, inserts stories, and asks for action.

Extract claims about these five categories of structural voice:

1. **Opening patterns** — what the creator does in the first ~10% of the transcript (hook, intro, preview of content, greeting).
2. **Closing patterns** — what the creator does in the last ~10% (summary, CTA, sign-off, sponsor pitch).
3. **Transition patterns** — how the creator moves between sections within a video. Verbal transitions only — claims about on-screen graphics, B-roll, or visual cues are NOT extractable from transcripts.
4. **Story insertion patterns** — when and how personal anecdotes appear in dense or instructional content (signal phrases, position within the video).
5. **CTA patterns** — what the creator asks of the audience (subscribe, comment, click, buy, follow elsewhere) and at what position in the video.

FOCUS: structural voice is about POSITION, ROLE, and TIMING — not the wording itself. There may be overlap with lexical voice patterns (e.g., a recurring sign-off phrase). When extracting structural claims, focus on the structural slot the phrase fills, not the phrase. The same sign-off may legitimately appear in both lexical and structural passes from different angles — that is fine.

POSITION FRAMING: transcripts have no timestamps. Use position within the transcript text as a proxy. "First ~10% of the transcript" counts as the opening; "last ~10%" counts as the closing. References to "the 60% mark" or "mid-video" mean roughly the corresponding portion of the transcript.

CITATION SEMANTICS: every claim must include at least one piece of evidence with a verbatim exact_quote from the transcript. The cited quote demonstrates ONE INSTANCE of the pattern, not proof that the pattern recurs. confidence_score should reflect HOW WIDESPREAD the pattern is across the bucket's transcripts, NOT the quality of the single citation. Pick the clearest exemplar quote you can find for each pattern.

GOOD claims are SPECIFIC and CITABLE:
- "Opens with a 30-second preview of three takeaways before the intro card"
- "Opens with an attention-grabbing question rather than a personal greeting"
- "Closes with 'see you later' followed by repeated 'bye-bye, bye-bye'"
- "Pitches a sponsor in the final ~60 seconds before the sign-off"
- "Marks each new section with 'tip number N' or 'the Nth one is'"
- "Bridges sections with 'now,' followed by a one-sentence preview of the next topic"
- "Drops a personal anecdote signaled by 'I remember when' around the 60% mark of educational videos"
- "Asks viewers to comment a specific question rather than a generic 'what do you think?'"
- "Pitches subscribing exactly once, in the last 30 seconds, never at the start"

BAD claims are GENERIC, VAGUE, or NON-EXTRACTABLE:
- "Has strong openings" — vague
- "Hooks viewers immediately" — generic, applies to any creator
- "Memorable closings" — vague
- "Smooth transitions" — vague, unfalsifiable
- "Uses storytelling" — generic
- "Engages viewers" — generic
- "Has CTAs" — vague
- Any claim about on-screen graphics, B-roll, visual cues, music, or video editing — NOT extractable from transcripts.

For each claim:
- claim_text: a one-sentence description of the structural pattern.
- evidence: at least one {video_id, exact_quote}. exact_quote MUST be a verbatim substring from the transcript exemplifying the pattern. video_id MUST be one of the IDs provided in the user message's <video id="..."> tags.
- confidence_score: 0.0 to 1.0, reflecting how widespread the pattern is across the bucket's transcripts.

Use the record_claims tool to return your findings. Aim for breadth — extract every distinctive structural pattern you can find. Synthesis across buckets will deduplicate later."""


STRUCTURAL_VOICE_SYNTHESIS_PROMPT = """You are merging structural voice claims that were extracted independently from multiple buckets of transcripts from the same creator. Many patterns will appear in multiple buckets phrased somewhat differently — structural abstractions naturally vary across buckets, and the same structural slot or role can be described in varying ways. Your job is to deduplicate them into canonical claims.

You will see input claims as a flat numbered list. For each canonical claim you produce, return:

- claim_text: the canonical phrasing of the pattern. Pick the clearest of the input phrasings, or write a better one.
- confidence_score: 0.0 to 1.0, reflecting how widespread the pattern is across all the input claims. A pattern appearing in many input claims should have higher confidence than one appearing only once.
- source_claim_indexes: a list of integer indexes into the input list that this canonical claim merges from. At least one index required.

Rules:
- Two claims describe the same pattern if they identify the same structural slot or role, even when the signal words or phrasings differ. BE PERMISSIVE in merging — for structural patterns, merging too aggressively is less harmful than splitting hair-distinctions. If two claims plausibly describe the same opening, closing, transition, story-insertion, or CTA pattern, merge them.
- Every canonical claim must have at least one source_claim_index. Do NOT invent canonical claims with no source.
- An input claim should typically map to exactly one canonical claim, or be dropped if it's too vague or weak to keep.
- You do NOT emit evidence — the orchestrator unions evidence from the source claims you reference. Just choose which source claims to merge.

Use the record_canonical_claims tool."""


def run_pass(
    creator_id: str,
    sb: Client,
    *,
    bucket_size: int = 10,
    llm_call: LlmCall | None = None,
) -> dict[str, Any]:
    """Run the structural voice extraction pass end-to-end.

    COST: ~$1.20-$1.40 per invocation against a typical channel. Real
    Anthropic API calls (claude-sonnet-4-6) — do NOT invoke casually
    expecting $0. For zero-cost testing, pass a stub `llm_call`.
    """
    return run_extraction_pass(
        creator_id,
        sb,
        pass_name=PASS_NAME,
        system_prompt=STRUCTURAL_VOICE_SYSTEM_PROMPT,
        synthesis_prompt=STRUCTURAL_VOICE_SYNTHESIS_PROMPT,
        bucket_size=bucket_size,
        llm_call=llm_call,
    )
