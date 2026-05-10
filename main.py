import asyncio
import os
import traceback
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
from niche_aggregation import run_niche_aggregation
from peer_pattern_extraction import run_peer_pattern_extraction
from peer_suggestion import run_peer_suggestion
from peer_sweep import run_peer_sweep

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


def _log_task_exception(label: str):
    """Returns a done-callback that logs unhandled exceptions from
    asyncio.create_task(...). Without this, exceptions raised inside the
    background task are dropped on the floor — the task object holds the
    exception but nothing ever calls .result() / .exception() on it. We've
    been bitten by this before in Phase 1a.
    """

    def _cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        tb = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        print(
            f"[main] {label} background task failed: "
            f"{type(exc).__name__}: {exc}\n{tb}",
            flush=True,
        )

    return _cb


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
    task = asyncio.create_task(
        asyncio.to_thread(
            ingest_youtube_channel,
            sb,
            body.analysis_id,
            body.channel_id,
            body.config,
        )
    )
    task.add_done_callback(
        _log_task_exception(f"ingest_youtube analysis_id={body.analysis_id}")
    )

    return {"status": "accepted", "analysis_id": body.analysis_id}


class SweepPeerRequest(BaseModel):
    peer_creator_id: str


@app.post("/sweep/peer", status_code=202)
async def sweep_peer(
    body: SweepPeerRequest,
    authorization: str | None = Header(default=None),
):
    """Background-task kickoff for one peer sweep. Same pattern as
    /ingest/youtube: returns 202 immediately, runs run_peer_sweep in a
    worker thread. The TS Inngest function (sweep-peer) polls
    peer_creators.peer_sweep_status until terminal.
    """
    verify_secret(authorization)
    sb = get_supabase()

    task = asyncio.create_task(
        asyncio.to_thread(run_peer_sweep, sb, body.peer_creator_id)
    )
    task.add_done_callback(
        _log_task_exception(
            f"run_peer_sweep peer_creator_id={body.peer_creator_id}"
        )
    )

    return {"status": "accepted", "peer_creator_id": body.peer_creator_id}


class AggregateNicheRequest(BaseModel):
    creator_id: str


@app.post("/aggregate-niche/creator", status_code=202)
async def aggregate_niche(
    body: AggregateNicheRequest,
    authorization: str | None = Header(default=None),
):
    """Background-task kickoff for one creator's niche aggregation. Same
    pattern as /extract-patterns/peer: 202 + asyncio task. The TS Inngest
    function (aggregate-niche) polls niche_pattern_aggregations for the
    latest row created at-or-after the API call anchor.
    """
    verify_secret(authorization)
    sb = get_supabase()

    task = asyncio.create_task(
        asyncio.to_thread(run_niche_aggregation, sb, body.creator_id)
    )
    task.add_done_callback(
        _log_task_exception(
            f"run_niche_aggregation creator_id={body.creator_id}"
        )
    )

    return {"status": "accepted", "creator_id": body.creator_id}


class SuggestPeersRequest(BaseModel):
    creator_id: str
    operator_context: str | None = None


@app.post("/suggest-peers/creator", status_code=202)
async def suggest_peers(
    body: SuggestPeersRequest,
    authorization: str | None = Header(default=None),
):
    """Background-task kickoff for one creator's peer-suggestion run.
    Same pattern as /sweep/peer and /extract-patterns/peer: 202 + asyncio
    task. The TS Inngest function (suggest-peers) polls peer_suggestions
    for rows with this creator_id and generated_at >= anchor.
    """
    verify_secret(authorization)
    sb = get_supabase()

    task = asyncio.create_task(
        asyncio.to_thread(
            run_peer_suggestion,
            sb,
            body.creator_id,
            body.operator_context,
        )
    )
    task.add_done_callback(
        _log_task_exception(
            f"run_peer_suggestion creator_id={body.creator_id}"
        )
    )

    return {"status": "accepted", "creator_id": body.creator_id}


class ExtractPeerPatternsRequest(BaseModel):
    peer_creator_id: str


@app.post("/extract-patterns/peer", status_code=202)
async def extract_peer_patterns(
    body: ExtractPeerPatternsRequest,
    authorization: str | None = Header(default=None),
):
    """Background-task kickoff for one peer's pattern extraction. Same
    pattern as /sweep/peer: returns 202 immediately, runs the extraction
    in a worker thread. The TS Inngest function (extract-peer-patterns)
    polls the latest peer_pattern_extractions row for terminal status.
    """
    verify_secret(authorization)
    sb = get_supabase()

    task = asyncio.create_task(
        asyncio.to_thread(
            run_peer_pattern_extraction, sb, body.peer_creator_id
        )
    )
    task.add_done_callback(
        _log_task_exception(
            f"run_peer_pattern_extraction peer_creator_id={body.peer_creator_id}"
        )
    )

    return {"status": "accepted", "peer_creator_id": body.peer_creator_id}


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
