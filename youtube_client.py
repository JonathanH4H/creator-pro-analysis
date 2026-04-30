import os
import re
from typing import Iterator

import httpx

YT_API = "https://www.googleapis.com/youtube/v3"


class YouTubeQuotaError(Exception):
    pass


def chunks(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class YouTubeClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.units_used = 0
        self._http = httpx.Client(timeout=30.0)
        # Dev-only debug toggle: when YOUTUBE_FORCE_QUOTA_AFTER_N is set in env,
        # the Nth+1 API call raises YouTubeQuotaError. No-op if env var unset.
        # Used to verify the quota_exceeded code path without burning real quota.
        force = os.environ.get("YOUTUBE_FORCE_QUOTA_AFTER_N")
        self._force_quota_remaining = int(force) if force else None

    def _request(self, path: str, params: dict, units: int = 1) -> dict:
        if self._force_quota_remaining is not None:
            if self._force_quota_remaining <= 0:
                raise YouTubeQuotaError(
                    "Forced quota exhaustion (YOUTUBE_FORCE_QUOTA_AFTER_N dev flag)"
                )
            self._force_quota_remaining -= 1

        params = {**params, "key": self.api_key}
        resp = self._http.get(f"{YT_API}/{path}", params=params)
        if resp.status_code == 403 and "quotaExceeded" in resp.text:
            raise YouTubeQuotaError(resp.text)
        resp.raise_for_status()
        self.units_used += units
        return resp.json()

    def fetch_video_comments(
        self, platform_video_id: str, max_results: int, channel_id: str
    ) -> list[dict]:
        # Top-level only, ordered by relevance. commentsDisabled is a normal
        # condition on YouTube — surface as empty list rather than error.
        try:
            data = self._request(
                "commentThreads",
                {
                    "part": "snippet",
                    "videoId": platform_video_id,
                    "order": "relevance",
                    "maxResults": min(max_results, 100),
                    "textFormat": "plainText",
                },
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403 and "commentsDisabled" in e.response.text:
                return []
            raise
        return [
            _normalize_comment(item, channel_id) for item in data.get("items", [])
        ]

    def get_uploads_playlist_id(self, channel_id: str) -> str:
        data = self._request("channels", {"part": "contentDetails", "id": channel_id})
        items = data.get("items", [])
        if not items:
            raise ValueError(f"Channel not found: {channel_id}")
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    def list_uploads(self, playlist_id: str) -> list[str]:
        ids: list[str] = []
        page_token: str | None = None
        while True:
            params: dict = {
                "part": "contentDetails",
                "playlistId": playlist_id,
                "maxResults": 50,
            }
            if page_token:
                params["pageToken"] = page_token
            data = self._request("playlistItems", params)
            ids.extend(item["contentDetails"]["videoId"] for item in data.get("items", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return ids

    def fetch_video_metadata(self, video_ids: list[str]) -> list[dict]:
        results: list[dict] = []
        for batch in chunks(video_ids, 50):
            data = self._request(
                "videos",
                {"part": "snippet,statistics,contentDetails", "id": ",".join(batch)},
            )
            for item in data.get("items", []):
                results.append(_normalize_video(item))
        return results


def _normalize_video(item: dict) -> dict:
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    content_details = item.get("contentDetails", {})
    thumbnails = snippet.get("thumbnails", {})
    return {
        "platform_video_id": item["id"],
        "title": snippet.get("title"),
        "description": snippet.get("description"),
        "published_at": snippet.get("publishedAt"),
        "duration_seconds": _parse_iso_duration(content_details.get("duration", "PT0S")),
        "view_count": int(statistics.get("viewCount", 0) or 0),
        "like_count": int(statistics.get("likeCount", 0) or 0),
        "comment_count": int(statistics.get("commentCount", 0) or 0),
        "thumbnail_url": (
            thumbnails.get("high", {}).get("url")
            or thumbnails.get("default", {}).get("url")
        ),
    }


def _normalize_comment(item: dict, channel_id: str) -> dict:
    top = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
    author_channel_id = (top.get("authorChannelId") or {}).get("value")
    return {
        "platform_comment_id": item.get("id"),
        "author": top.get("authorDisplayName"),
        "text": top.get("textDisplay"),
        "like_count": int(top.get("likeCount", 0) or 0),
        "is_creator_reply": author_channel_id == channel_id,
        "published_at": top.get("publishedAt"),
    }


def _parse_iso_duration(iso: str) -> int:
    match = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", iso or "")
    if not match:
        return 0
    h, m, s = (int(g) if g else 0 for g in match.groups())
    return h * 3600 + m * 60 + s
