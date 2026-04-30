import os
from datetime import datetime, timezone
from typing import Any

import httpx
from supabase import Client

from transcripts import transcribe_video
from youtube_client import YouTubeClient, YouTubeQuotaError, chunks


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit_progress(
    sb: Client,
    analysis_id: str,
    *,
    stage: str,
    current: int = 0,
    total: int = 0,
    errors: list | None = None,
):
    progress = {
        "stage": stage,
        "current": current,
        "total": total,
        "started_at": _utcnow_iso(),
        "errors": errors or [],
    }
    sb.table("analyses").update({"pipeline_progress": progress}).eq(
        "id", analysis_id
    ).execute()


def _persist_units(sb: Client, analysis_id: str, units: int):
    sb.table("analyses").update({"units_used": units}).eq("id", analysis_id).execute()


def ingest_youtube_channel(
    sb: Client,
    analysis_id: str,
    channel_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    api_key = os.environ["YOUTUBE_API_KEY"]
    client = YouTubeClient(api_key)

    pa = (
        sb.table("platform_accounts")
        .select("id")
        .eq("platform", "youtube")
        .eq("platform_account_id", channel_id)
        .single()
        .execute()
    )
    platform_account_id = pa.data["id"]

    sb.table("analyses").update({"status": "running"}).eq("id", analysis_id).execute()

    try:
        # Stage 1: list uploads
        _emit_progress(sb, analysis_id, stage="listing_videos")
        playlist_id = client.get_uploads_playlist_id(channel_id)
        all_video_ids = client.list_uploads(playlist_id)
        _emit_progress(
            sb, analysis_id, stage="listing_videos",
            current=len(all_video_ids), total=len(all_video_ids),
        )
        _persist_units(sb, analysis_id, client.units_used)

        # Stage 2: metadata for ALL uploads (need view counts to identify top-by-views)
        _emit_progress(sb, analysis_id, stage="fetching_metadata", current=0, total=len(all_video_ids))
        all_videos: list[dict] = []
        for batch in chunks(all_video_ids, 50):
            metadata = client.fetch_video_metadata(batch)
            all_videos.extend(metadata)
            _emit_progress(
                sb, analysis_id, stage="fetching_metadata",
                current=len(all_videos), total=len(all_video_ids),
            )
            _persist_units(sb, analysis_id, client.units_used)

        # Shorts filter
        if not config.get("include_shorts", False):
            all_videos = [v for v in all_videos if (v.get("duration_seconds") or 0) >= 60]

        # Recent + top selection (deduped union)
        recent_ids = _select_recent(all_videos, config.get("recent_count", 50))
        top_ids = _select_top(all_videos, config.get("top_count", 50))
        selected = recent_ids | top_ids
        videos_to_ingest = [v for v in all_videos if v["platform_video_id"] in selected]
        total_to_ingest = len(videos_to_ingest)

        # Re-ingestion partition. Stage 3 (upsert) runs on every selected video
        # so view/like/comment counts on existing rows get refreshed. Stages 4,
        # 5, 6 (thumbnails, transcripts, comments) run on new videos only —
        # these are expensive (Whisper $$, comments quota) and the data on
        # existing rows is either unchanged (transcripts/thumbnails) or, per
        # Phase 1a constraint, intentionally not refreshed (comments).
        existing_pvids = _fetch_existing_pvids(
            sb, platform_account_id, [v["platform_video_id"] for v in videos_to_ingest]
        )
        new_videos = [
            v for v in videos_to_ingest if v["platform_video_id"] not in existing_pvids
        ]
        total_new = len(new_videos)

        # Stage 3: write videos
        _emit_progress(sb, analysis_id, stage="writing_videos", current=0, total=total_to_ingest)
        for i, v in enumerate(videos_to_ingest):
            _upsert_video(sb, platform_account_id, v)
            if (i + 1) % 10 == 0 or (i + 1) == total_to_ingest:
                _emit_progress(
                    sb, analysis_id, stage="writing_videos",
                    current=i + 1, total=total_to_ingest,
                )

        # Stage 4: thumbnails (bucket is public, service role bypasses RLS for write)
        _emit_progress(sb, analysis_id, stage="downloading_thumbnails", current=0, total=total_new)
        for i, v in enumerate(new_videos):
            path = _upload_thumbnail(sb, platform_account_id, v)
            if path:
                sb.table("videos").update({"thumbnail_storage_path": path}).eq(
                    "platform_account_id", platform_account_id
                ).eq("platform_video_id", v["platform_video_id"]).execute()
            if (i + 1) % 10 == 0 or (i + 1) == total_new:
                _emit_progress(
                    sb, analysis_id, stage="downloading_thumbnails",
                    current=i + 1, total=total_new,
                )

        # Stage 5: transcripts. Whisper is the primary path (default true);
        # youtube-transcript-api is the toggle-off cost-conscious override.
        # `whisper_fallback` is the vestigial config-key name from chunk 1
        # (see lib/validation/analysis.ts comment).
        #
        # Sequential by design — a parallelism shim (asyncio.gather + semaphore
        # of 5) is a ~5x speedup if needed, isolated to this loop, no
        # architectural impact. Defer until a real reason emerges (operator
        # complaint, demo dragging, downstream feature requirement).
        use_whisper = bool(config.get("whisper_fallback", True))
        _emit_progress(
            sb, analysis_id, stage="fetching_transcripts",
            current=0, total=total_new,
        )
        for i, v in enumerate(new_videos):
            text, source = transcribe_video(
                v["platform_video_id"], use_whisper=use_whisper
            )
            sb.table("videos").update(
                {"transcript": text, "transcript_source": source}
            ).eq("platform_account_id", platform_account_id).eq(
                "platform_video_id", v["platform_video_id"]
            ).execute()
            if (i + 1) % 10 == 0 or (i + 1) == total_new:
                _emit_progress(
                    sb, analysis_id, stage="fetching_transcripts",
                    current=i + 1, total=total_new,
                )

        # Stage 6: comments. Top-level only, ordered by relevance. New videos
        # only — Phase 1a constraint per spec chunk 6: existing videos do not
        # get their comments refreshed. Idempotent upsert dedupes within a run.
        # commentsDisabled is empty-list-continue, not failure.
        comments_per_video = int(config.get("comments_per_video", 50))
        video_uuid_by_pvid = _fetch_video_uuid_map(
            sb, platform_account_id, [v["platform_video_id"] for v in new_videos]
        )
        _emit_progress(
            sb, analysis_id, stage="fetching_comments",
            current=0, total=total_new,
        )
        for i, v in enumerate(new_videos):
            video_uuid = video_uuid_by_pvid[v["platform_video_id"]]
            comments = client.fetch_video_comments(
                v["platform_video_id"], comments_per_video, channel_id
            )
            for c in comments:
                _upsert_comment(sb, video_uuid, c)
            if (i + 1) % 10 == 0 or (i + 1) == total_new:
                _emit_progress(
                    sb, analysis_id, stage="fetching_comments",
                    current=i + 1, total=total_new,
                )
                _persist_units(sb, analysis_id, client.units_used)

        sb.table("analyses").update({
            "status": "completed",
            "completed_at": _utcnow_iso(),
            "units_used": client.units_used,
        }).eq("id", analysis_id).execute()

        return {
            "ok": True,
            "video_count": total_to_ingest,
            "units_used": client.units_used,
        }

    except YouTubeQuotaError as e:
        sb.table("analyses").update({
            "status": "quota_exceeded",
            "error_message": str(e)[:500],
            "units_used": client.units_used,
        }).eq("id", analysis_id).execute()
        return {"ok": False, "error": "quota_exceeded", "units_used": client.units_used}
    except Exception as e:
        # Production failures should be surfaced via real error monitoring —
        # see PRE_LAUNCH_TODO.md "Add error monitoring (Sentry, …)".
        sb.table("analyses").update({
            "status": "failed",
            "error_message": str(e)[:500],
            "units_used": client.units_used,
        }).eq("id", analysis_id).execute()
        raise


def _select_recent(videos: list[dict], count: int | str) -> set[str]:
    if count == "all":
        return {v["platform_video_id"] for v in videos}
    return {v["platform_video_id"] for v in videos[: int(count)]}


def _select_top(videos: list[dict], count: int | str) -> set[str]:
    if count == "all":
        return {v["platform_video_id"] for v in videos}
    sorted_by_views = sorted(videos, key=lambda v: v.get("view_count") or 0, reverse=True)
    return {v["platform_video_id"] for v in sorted_by_views[: int(count)]}


def _upsert_video(sb: Client, platform_account_id: str, video: dict):
    payload = {
        "platform_account_id": platform_account_id,
        "platform_video_id": video["platform_video_id"],
        "title": video.get("title"),
        "description": video.get("description"),
        "published_at": video.get("published_at"),
        "duration_seconds": video.get("duration_seconds"),
        "view_count": video.get("view_count"),
        "like_count": video.get("like_count"),
        "comment_count": video.get("comment_count"),
        "thumbnail_url": video.get("thumbnail_url"),
    }
    sb.table("videos").upsert(
        payload, on_conflict="platform_account_id,platform_video_id"
    ).execute()


def _fetch_existing_pvids(
    sb: Client, platform_account_id: str, platform_video_ids: list[str]
) -> set[str]:
    if not platform_video_ids:
        return set()
    rows = (
        sb.table("videos")
        .select("platform_video_id")
        .eq("platform_account_id", platform_account_id)
        .in_("platform_video_id", platform_video_ids)
        .execute()
    )
    return {r["platform_video_id"] for r in rows.data}


def _fetch_video_uuid_map(
    sb: Client, platform_account_id: str, platform_video_ids: list[str]
) -> dict[str, str]:
    rows = (
        sb.table("videos")
        .select("id, platform_video_id")
        .eq("platform_account_id", platform_account_id)
        .in_("platform_video_id", platform_video_ids)
        .execute()
    )
    return {r["platform_video_id"]: r["id"] for r in rows.data}


def _upsert_comment(sb: Client, video_uuid: str, comment: dict):
    payload = {"video_id": video_uuid, **comment}
    sb.table("comments").upsert(
        payload, on_conflict="video_id,platform_comment_id"
    ).execute()


def _upload_thumbnail(sb: Client, platform_account_id: str, video: dict) -> str | None:
    url = video.get("thumbnail_url")
    if not url:
        return None
    try:
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        path = f"{platform_account_id}/{video['platform_video_id']}.jpg"
        sb.storage.from_("thumbnails").upload(
            path,
            resp.content,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
        return path
    except Exception:
        # Thumbnail failures are tolerated to avoid blocking ingestion of an
        # otherwise-good video row. Caveat: chunk 6 re-ingestion skips this
        # stage for existing videos, so a silently-failed thumbnail leaves
        # the row permanently without thumbnail_storage_path. See
        # PRE_LAUNCH_TODO "Reliability" for fix options.
        return None
