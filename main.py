import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

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


@app.post("/ingest/youtube")
def ingest_youtube(
    body: IngestYouTubeRequest,
    authorization: str | None = Header(default=None),
):
    verify_secret(authorization)

    # Phase 1a Chunk 2: stub. Mark the run as running and write one progress
    # event. Real ingestion (video listing, metadata, transcripts, comments)
    # lands in chunks 3-5.
    sb = get_supabase()
    progress = {
        "stage": "started",
        "current": 0,
        "total": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "errors": [],
    }
    sb.table("analyses").update(
        {"status": "running", "pipeline_progress": progress}
    ).eq("id", body.analysis_id).execute()

    return {"ok": True, "analysis_id": body.analysis_id}
