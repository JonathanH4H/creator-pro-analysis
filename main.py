import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

import avatar_profile
import lexical_voice
import performance_profile
import structural_voice
import topical_voice
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


PASS_RUNNERS = {
    "lexical_voice": lexical_voice.run_pass,
    "structural_voice": structural_voice.run_pass,
    "topical_voice": topical_voice.run_pass,
    "avatar": avatar_profile.run_pass,
    "performance": performance_profile.run_pass,
}


class ExtractDnaPassRequest(BaseModel):
    analysis_id: str
    creator_id: str
    pass_name: str


@app.post("/extract/dna-pass")
def extract_dna_pass(
    body: ExtractDnaPassRequest,
    authorization: str | None = Header(default=None),
):
    """Run a single DNA extraction pass synchronously and return its result.

    Inngest's extract-dna function calls this once per pass (lexical,
    structural, topical) sequentially. Pass-level retries happen at the
    Inngest step layer.

    Status code semantics for retry classification:
      - 5xx        → transient (Inngest retries)
      - 4xx (400)  → permanent (NonRetriableError on the TS side)
      - 401        → auth misconfig (permanent until env fixed)
    """
    verify_secret(authorization)
    sb = get_supabase()

    runner = PASS_RUNNERS.get(body.pass_name)
    if runner is None:
        raise HTTPException(
            400,
            f"Unknown pass_name '{body.pass_name}'. "
            f"Valid: {sorted(PASS_RUNNERS.keys())}",
        )

    try:
        result = runner(body.creator_id, sb)
    except ValueError as e:
        # Bad input — no transcripts, no platform_account, etc. Permanent.
        raise HTTPException(400, str(e))
    except Exception as e:
        # Anthropic transport / runtime errors. Treat as transient by default
        # so Inngest retries; persistent failures will exhaust retries and
        # surface as dna_pass_runs.status='failed'.
        raise HTTPException(500, f"{type(e).__name__}: {e}")

    return result
