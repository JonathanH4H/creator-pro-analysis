import os
import traceback
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any

from supabase import Client

from transcripts import transcribe_video
from youtube_client import YouTubeClient, chunks

# Whisper-1 list price as of 1c.3. Source of truth for last_sweep_cost_usd.
WHISPER_USD_PER_MINUTE = 0.006

# Sweep window + outlier threshold. Stored on peer_creators per sweep so a
# downstream consumer can audit how a given outlier set was computed.
WINDOW_DAYS = 365
THRESHOLD_RATIO = 2.0

# Mirrors ingest._upsert_transcript's rank pattern. Whisper > yta > nothing.
# Equal-rank attempts overwrite; lower-rank attempts preserve the better one.
TRANSCRIPT_RANK: dict[str | None, int] = {
    None: 0,
    "unavailable": 0,
    "youtube-transcript-api": 1,
    "whisper": 2,
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit_progress(
    sb: Client,
    peer_creator_id: str,
    *,
    stage: str,
    current: int = 0,
    total: int = 0,
):
    progress = {
        "stage": stage,
        "current": current,
        "total": total,
        "updated_at": _utcnow_iso(),
    }
    sb.table("peer_creators").update({"sweep_progress": progress}).eq(
        "id", peer_creator_id
    ).execute()


def _should_write_transcript(
    existing_source: str | None, new_source: str | None
) -> bool:
    return TRANSCRIPT_RANK.get(new_source, 0) >= TRANSCRIPT_RANK.get(
        existing_source, 0
    )


def run_peer_sweep(sb: Client, peer_creator_id: str) -> dict[str, Any]:
    """Sweep one peer: enumerate uploads, identify outliers in the 365-day
    window (view_count >= 2 * peer median), upsert outlier rows, fetch
    transcripts (rank-aware preserve so we don't re-pay Whisper for cached
    transcripts).

    On any uncaught exception the peer_creators row is marked failed with
    the error message, then the exception is re-raised so the asyncio task
    surfaces it in logs.
    """
    print(
        f"[peer_sweep] {peer_creator_id}: run_peer_sweep called", flush=True
    )

    api_key = os.environ["YOUTUBE_API_KEY"]
    client = YouTubeClient(api_key)

    peer_row = (
        sb.table("peer_creators")
        .select("id, platform_account_id")
        .eq("id", peer_creator_id)
        .single()
        .execute()
    ).data
    if not peer_row:
        raise ValueError(f"peer_creators row not found: {peer_creator_id}")
    channel_id = peer_row["platform_account_id"]
    print(
        f"[peer_sweep] {peer_creator_id}: channel_id={channel_id}", flush=True
    )

    # Reset sweep state at the start so prior failure messages / progress
    # don't bleed into the new run.
    sb.table("peer_creators").update(
        {
            "peer_sweep_status": "sweeping",
            "last_sweep_error": None,
            "sweep_progress": {
                "stage": "starting",
                "current": 0,
                "total": 0,
                "updated_at": _utcnow_iso(),
            },
        }
    ).eq("id", peer_creator_id).execute()

    try:
        # Stage 1: enumerate uploads via the channel's uploads playlist
        # (1 unit per page — much cheaper than search.list at 100 units).
        _emit_progress(sb, peer_creator_id, stage="listing_videos")
        playlist_id = client.get_uploads_playlist_id(channel_id)
        print(
            f"[peer_sweep] {peer_creator_id}: get_uploads_playlist_id → {playlist_id}",
            flush=True,
        )
        all_video_ids = client.list_uploads(playlist_id)
        print(
            f"[peer_sweep] {peer_creator_id}: list_uploads → {len(all_video_ids)} video IDs",
            flush=True,
        )
        _emit_progress(
            sb,
            peer_creator_id,
            stage="listing_videos",
            current=len(all_video_ids),
            total=len(all_video_ids),
        )

        # Stage 2: metadata for all uploads. We need view_count to compute
        # median + identify outliers; we can't filter by date until we have
        # published_at. The uploads playlist returns reverse-chronological,
        # so we could short-circuit once we cross the 365-day cutoff to
        # avoid paying for older metadata — defer that optimization until
        # we see peers with truly massive back catalogs.
        _emit_progress(
            sb,
            peer_creator_id,
            stage="fetching_metadata",
            current=0,
            total=len(all_video_ids),
        )
        all_videos: list[dict] = []
        for batch in chunks(all_video_ids, 50):
            metadata = client.fetch_video_metadata(batch)
            all_videos.extend(metadata)
            _emit_progress(
                sb,
                peer_creator_id,
                stage="fetching_metadata",
                current=len(all_videos),
                total=len(all_video_ids),
            )
        print(
            f"[peer_sweep] {peer_creator_id}: fetch_video_metadata → {len(all_videos)} videos",
            flush=True,
        )
        # First 3 published_at values, before any filtering. If date parsing
        # is dropping rows, this makes the input format obvious vs. inferring
        # from a "0 in window" count alone.
        sample_published_at = [v.get("published_at") for v in all_videos[:3]]
        print(
            f"[peer_sweep] {peer_creator_id}: sample published_at: {sample_published_at}",
            flush=True,
        )

        # Stage 3: window filter + median + outlier identification.
        cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
        in_window: list[dict] = []
        for v in all_videos:
            pub_at = v.get("published_at")
            if not pub_at:
                continue
            if datetime.fromisoformat(pub_at.replace("Z", "+00:00")) >= cutoff:
                in_window.append(v)
        print(
            f"[peer_sweep] {peer_creator_id}: window filter ({WINDOW_DAYS}d) → {len(in_window)} in window",
            flush=True,
        )

        view_counts = [v.get("view_count") or 0 for v in in_window]
        peer_median = int(median(view_counts)) if view_counts else None

        if peer_median and peer_median > 0:
            threshold = peer_median * THRESHOLD_RATIO
            outliers = [
                v for v in in_window if (v.get("view_count") or 0) >= threshold
            ]
        else:
            outliers = []
        threshold_str = (
            f"{peer_median * THRESHOLD_RATIO}" if peer_median else "n/a"
        )
        print(
            f"[peer_sweep] {peer_creator_id}: median={peer_median}, threshold={threshold_str}, outliers={len(outliers)}",
            flush=True,
        )

        # Stage 4: clear outlier flag on every existing peer_videos row so
        # the result reflects this sweep only. Rows that were outliers last
        # time but aren't now stay in the table (with their transcript) but
        # flip is_outlier=false.
        sb.table("peer_videos").update({"is_outlier": False}).eq(
            "peer_creator_id", peer_creator_id
        ).execute()

        # Stage 5: upsert outliers + fetch transcripts (rank-aware preserve).
        # Cost accounting: only count Whisper minutes for videos we actually
        # transcribed in this sweep (skipping cached whisper hits saves $$).
        _emit_progress(
            sb,
            peer_creator_id,
            stage="ingesting_outliers",
            current=0,
            total=len(outliers),
        )
        whisper_cost_usd = 0.0
        for i, v in enumerate(outliers):
            pvid = v["platform_video_id"]
            view_count = v.get("view_count") or 0
            outlier_score = (
                round(view_count / peer_median, 4) if peer_median else None
            )

            # Upsert metadata first. Payload omits transcript columns so an
            # existing transcript on this row is preserved through the
            # upsert. supabase-py upsert generates ON CONFLICT DO UPDATE
            # SET only for the columns in the payload.
            sb.table("peer_videos").upsert(
                {
                    "peer_creator_id": peer_creator_id,
                    "platform_video_id": pvid,
                    "title": v.get("title"),
                    "description": v.get("description"),
                    "duration_seconds": v.get("duration_seconds"),
                    "view_count": view_count,
                    "like_count": v.get("like_count"),
                    "comment_count": v.get("comment_count"),
                    "published_at": v.get("published_at"),
                    "thumbnail_url": v.get("thumbnail_url"),
                    "outlier_score": outlier_score,
                    "is_outlier": True,
                },
                on_conflict="peer_creator_id,platform_video_id",
            ).execute()

            existing = (
                sb.table("peer_videos")
                .select("transcript_source")
                .eq("peer_creator_id", peer_creator_id)
                .eq("platform_video_id", pvid)
                .single()
                .execute()
            ).data
            existing_source = (existing or {}).get("transcript_source")
            print(
                f"[peer_sweep] {peer_creator_id}: outlier {i+1}/{len(outliers)} {pvid}: existing_source={existing_source}",
                flush=True,
            )

            # If we already have a whisper transcript, skip — re-fetching
            # would cost real money for no gain.
            if (
                TRANSCRIPT_RANK.get(existing_source, 0)
                >= TRANSCRIPT_RANK["whisper"]
            ):
                print(
                    f"[peer_sweep] {peer_creator_id}: outlier {pvid}: transcript=cached-{existing_source}-preserved, whisper_cost+=$0.0000",
                    flush=True,
                )
                _emit_progress(
                    sb,
                    peer_creator_id,
                    stage="ingesting_outliers",
                    current=i + 1,
                    total=len(outliers),
                )
                continue

            text, source = transcribe_video(pvid, use_whisper=True)
            if _should_write_transcript(existing_source, source):
                sb.table("peer_videos").update(
                    {"transcript": text, "transcript_source": source}
                ).eq("peer_creator_id", peer_creator_id).eq(
                    "platform_video_id", pvid
                ).execute()
            cost_added = 0.0
            if source == "whisper" and v.get("duration_seconds"):
                cost_added = (
                    v["duration_seconds"] / 60
                ) * WHISPER_USD_PER_MINUTE
                whisper_cost_usd += cost_added
            print(
                f"[peer_sweep] {peer_creator_id}: outlier {pvid}: transcript={source}, whisper_cost+=${cost_added:.4f}",
                flush=True,
            )

            _emit_progress(
                sb,
                peer_creator_id,
                stage="ingesting_outliers",
                current=i + 1,
                total=len(outliers),
            )

        # Terminal write. last_outlier_sweep_at is the audit timestamp; the
        # window/threshold pair lets a future debugger reconstruct what
        # constituted an outlier on this run.
        sb.table("peer_creators").update(
            {
                "peer_sweep_status": "complete",
                "last_outlier_sweep_at": _utcnow_iso(),
                "median_view_count": peer_median,
                "last_sweep_cost_usd": round(whisper_cost_usd, 4),
                "last_sweep_outlier_count": len(outliers),
                "last_sweep_window_days": WINDOW_DAYS,
                "last_sweep_threshold_ratio": THRESHOLD_RATIO,
                "last_sweep_error": None,
                "sweep_progress": {
                    "stage": "complete",
                    "current": len(outliers),
                    "total": len(outliers),
                    "updated_at": _utcnow_iso(),
                },
            }
        ).eq("id", peer_creator_id).execute()
        print(
            f"[peer_sweep] {peer_creator_id}: complete: {len(outliers)} outliers, ${whisper_cost_usd:.2f} total, median={peer_median}",
            flush=True,
        )

        return {
            "ok": True,
            "outliers": len(outliers),
            "cost_usd": round(whisper_cost_usd, 4),
            "median": peer_median,
            "videos_in_window": len(in_window),
        }

    except Exception as e:
        # Log the full traceback BEFORE the DB write so even if the write
        # fails we still see what went wrong in Railway logs.
        tb = traceback.format_exc()
        print(
            f"[peer_sweep] {peer_creator_id}: FAILED with {type(e).__name__}: {e}\n{tb}",
            flush=True,
        )
        sb.table("peer_creators").update(
            {
                "peer_sweep_status": "failed",
                "last_sweep_error": str(e)[:500],
                "sweep_progress": {
                    "stage": "failed",
                    "updated_at": _utcnow_iso(),
                    "error": str(e)[:200],
                },
            }
        ).eq("id", peer_creator_id).execute()
        raise
