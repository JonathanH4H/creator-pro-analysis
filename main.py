import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Creator Pro Analysis Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before launch — see PRE_LAUNCH_TODO.md
    allow_methods=["*"],
    allow_headers=["*"],
)

SHARED_SECRET = os.environ.get("ANALYSIS_SERVICE_SECRET")


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
