import os
from datetime import datetime, timezone
from typing import Any

import httpx
from supabase import Client

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
        _emit_progress(sb, analysis_id, stage="downloading_thumbnails", current=0, total=total_to_ingest)
        for i, v in enumerate(videos_to_ingest):
            path = _upload_thumbnail(sb, platform_account_id, v)
            if path:
                sb.table("videos").update({"thumbnail_storage_path": path}).eq(
                    "platform_account_id", platform_account_id
                ).eq("platform_video_id", v["platform_video_id"]).execute()
            if (i + 1) % 10 == 0 or (i + 1) == total_to_ingest:
                _emit_progress(
                    sb, analysis_id, stage="downloading_thumbnails",
                    current=i + 1, total=total_to_ingest,
                )

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
        # Thumbnail failures are tolerable: video row is created without
        # thumbnail_storage_path; a re-run can populate it later.
        return None
