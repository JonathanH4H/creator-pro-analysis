"""Avatar (audience) DNA pass (Phase 1b.6).

Extracts audience traits from comments. Buckets are per-video — all
comments on a video go into one bucket. Bucket count = number of videos
with ≥MIN_COMMENTS_PER_VIDEO ingested comments.

CRITICAL framing: avatar claims are about the AUDIENCE (the people
behind the comments), not about the comments themselves. The cited
quote is evidence pointing at an audience trait.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from supabase import Client

from dna import (
    COMMENT_TEXT,
    LlmCall,
    BucketLoader,
    run_extraction_pass,
)


PASS_NAME = "avatar"
MIN_COMMENTS_PER_VIDEO = 5


AVATAR_SYSTEM_PROMPT = """You are an expert in audience analysis. You will be given a set of YouTube comments on a single video from a creator. Your task is to identify avatar (audience) patterns — what kind of viewer this creator attracts, inferred from how viewers describe themselves and engage in the comments.

CRITICAL: avatar claims are about the AUDIENCE (the people behind the comments), not about the comments themselves. "Many commenters use exclamation marks" is a claim about COMMENT TEXT — wrong. "Audience skews toward students preparing for university entrance exams" is a claim about the AUDIENCE — right. The cited quote is evidence pointing at the audience trait, not the trait being commented-upon.

Extract claims about these five categories of avatar:

1. **Demographic signals** — age range, life stage, occupation/role (students, professionals, retirees, parents). Inferred from how commenters describe themselves.
2. **Interest signals** — what topics commenters bring up unprompted, what they care about beyond the creator's content.
3. **Value signals** — what they appreciate or criticize about the creator's content, sponsorships, style, perspective.
4. **Engagement patterns** — how viewers engage with the creator (asking questions, sharing experiences, disagreeing politely, requesting topics, expressing gratitude).
5. **Language register** — formal vs casual, jargon use, native vs non-native English signals (calques, ESL grammar, second-language formality).

GOOD claims are SPECIFIC and CITABLE:
- "Audience skews toward students preparing for medical school admissions"
- "Audience commonly asks for follow-up content on specific subtopics rather than general appreciation"
- "Audience includes a sizable contingent of non-native English speakers, often citing English as a second language they're learning alongside the creator's subject matter"
- "Audience values the creator's lack of overt sponsorships and frequently comments on this"

BAD claims (about COMMENTS, not AUDIENCE) — these are wrong even if literally true:
- "Many commenters use exclamation marks" — about comment text, not audience
- "Comments are mostly short" — about comment length, not audience
- "Commenters often start with 'Hi'" — about comment phrasing, not audience trait

BAD claims (vague/generic):
- "Audience is engaged" — vague
- "Audience likes the content" — generic
- "Diverse audience" — unfalsifiable

For each claim:
- claim_text: a one-sentence description of the audience trait. Phrase as a statement about the audience, not about the comments.
- evidence: at least one {comment_id, exact_quote}. exact_quote MUST be a verbatim substring from a comment. comment_id MUST be one of the IDs provided in the user message's <comment id="..."> tags.
- confidence_score: 0.0 to 1.0, reflecting how widespread the trait is across the bucket's comments.

Use the record_claims tool. Aim for breadth — extract every distinctive audience trait you can find. Synthesis across buckets will deduplicate later."""


AVATAR_SYNTHESIS_PROMPT = """You are merging avatar (audience) claims that were extracted independently from multiple buckets of comments from the same creator. Each bucket covers comments on a single video. The same audience trait may surface in multiple buckets phrased differently. Your job is to deduplicate them into canonical claims.

You will see input claims as a flat numbered list. For each canonical claim you produce, return:

- claim_text: the canonical phrasing of the audience trait. Pick the clearest of the input phrasings, or write a better one. Stay framed as a statement about the audience, not about the comments.
- confidence_score: 0.0 to 1.0, reflecting how widespread the trait is across all the input claims. A trait appearing in many input claims should have higher confidence.
- source_claim_indexes: a list of integer indexes into the input list that this canonical claim merges from.

Rules:
- Two claims describe the same trait if they identify the same audience characteristic, even when phrased differently. BE PERMISSIVE in merging — different videos surface different facets of the same audience. Merging too aggressively is less harmful than splitting hair-distinctions.
- Every canonical claim must have at least one source_claim_index. Do NOT invent canonical claims with no source.
- An input claim should typically map to exactly one canonical claim, or be dropped if it's too vague to keep.
- You do NOT emit evidence — the orchestrator unions evidence from the source claims you reference.

Use the record_canonical_claims tool."""


def comment_bucket_loader(
    min_comments_per_video: int = MIN_COMMENTS_PER_VIDEO,
) -> BucketLoader:
    """Returns a bucket loader that groups comments by video and skips
    videos with fewer than min_comments_per_video ingested comments.

    Items have shape {"id": comment_uuid, "source_text": comment_text}.
    Each bucket is one video's comments.
    """

    def _load(creator_id: str, sb: Client) -> list[list[dict]]:
        pa_rows = (
            sb.table("platform_accounts")
            .select("id")
            .eq("creator_id", creator_id)
            .execute()
        ).data
        if not pa_rows:
            raise ValueError(
                f"No platform_accounts for creator_id={creator_id}"
            )
        pa_ids = [r["id"] for r in pa_rows]

        videos = (
            sb.table("videos")
            .select("id")
            .in_("platform_account_id", pa_ids)
            .execute()
        ).data
        video_ids = [v["id"] for v in videos]
        if not video_ids:
            raise ValueError(f"No videos for creator_id={creator_id}")

        comments = (
            sb.table("comments")
            .select("id, video_id, text")
            .in_("video_id", video_ids)
            .execute()
        ).data
        by_video: dict[str, list[dict]] = defaultdict(list)
        for c in comments:
            text = c.get("text")
            if not text:
                continue
            by_video[c["video_id"]].append(
                {"id": c["id"], "source_text": text}
            )

        buckets = [
            items
            for items in by_video.values()
            if len(items) >= min_comments_per_video
        ]
        if not buckets:
            raise ValueError(
                f"No videos with ≥{min_comments_per_video} ingested comments "
                f"for creator_id={creator_id}"
            )
        return buckets

    return _load


def run_pass(
    creator_id: str,
    sb: Client,
    *,
    min_comments_per_video: int = MIN_COMMENTS_PER_VIDEO,
    llm_call: LlmCall | None = None,
) -> dict[str, Any]:
    """Run the avatar extraction pass end-to-end.

    COST: ~$1-2 per invocation depending on comment volume. Comments are
    short (typically <500 chars each), but bucket count scales with the
    number of qualifying videos. Real Anthropic API calls — do NOT invoke
    casually expecting $0. For zero-cost testing, pass a stub `llm_call`.
    """
    return run_extraction_pass(
        creator_id,
        sb,
        pass_name=PASS_NAME,
        system_prompt=AVATAR_SYSTEM_PROMPT,
        synthesis_prompt=AVATAR_SYNTHESIS_PROMPT,
        bucket_loader=comment_bucket_loader(min_comments_per_video),
        evidence_source=COMMENT_TEXT,
        llm_call=llm_call,
    )
