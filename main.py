import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

from ingest import ingest_youtube_channel

load_dotenv()
app = FastAPI(title="Creator Pro Analysis Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before launch — see PRE_LAUNCH_TODO.md
    allow_methods=["*"],
    allow_headers=["*"],
)

SHARED_SECRET = os.environ.get("ANALYSIS_SERVICE_SECRET")

_supabase: Client | None = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.environ.get("SUPABASE_URL") or os.environ.get(
            "NEXT_PUBLIC_SUPABASE_URL"
        )
        if not url:
            raise RuntimeError(
                "SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) must be set"
            )
        _supabase = create_client(url, os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    return _supabase


def verify_secret(authorization: str | None):
    if not SHARED_SECRET:
        raise HTTPException(500, "Service misconfigured")
    if authorization != f"Bearer {SHARED_SECRET}":
        raise HTTPException(401, "Unauthorized")


@app.get("/")
def root():
    return {"service": "creator-pro-analysis", "status": "ok"}


@app.get("/health")
def health(authorization: str | None = Header(default=None)):
    verify_secret(authorization)
    return {"status": "healthy"}


class IngestYouTubeRequest(BaseModel):
    analysis_id: str
    channel_id: str
    config: dict[str, Any]


@app.post("/ingest/youtube", status_code=202)
async def ingest_youtube(
    body: IngestYouTubeRequest,
    authorization: str | None = Header(default=None),
):
    verify_secret(authorization)
    sb = get_supabase()

    # Pragmatic background pattern for internal Phase 1a use. ingest_youtube_channel
    # is sync (supabase-py, yt-dlp, youtube-transcript-api are all sync), so we run
    # it in a worker thread via asyncio.to_thread and let the request return 202.
    #
    # Limitation: tasks live in the uvicorn worker process. uvicorn restart kills
    # in-flight ingestions (no resumption; analyses row stays at 'running' until
    # Inngest's MAX_POLLS timeout marks the function failed). Migrate to a real
    # task queue (Celery / Arq / RQ) before scaling beyond internal — see
    # PRE_LAUNCH_TODO.md "Reliability".
    asyncio.create_task(
        asyncio.to_thread(
            ingest_youtube_channel,
            sb,
            body.analysis_id,
            body.channel_id,
            body.config,
        )
    )

    return {"status": "accepted", "analysis_id": body.analysis_id}
