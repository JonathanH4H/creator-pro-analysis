import os
import tempfile

import yt_dlp
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi


class WhisperUnavailableError(Exception):
    pass


def fetch_transcript_native(video_id: str) -> str | None:
    """youtube-transcript-api path. Reachable only when use_whisper=False
    (cost-conscious override). Captures only ~5% of videos in practice
    in our environment, which is why Whisper is the primary path."""
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)
        return " ".join(snippet.text for snippet in fetched)
    except Exception as e:
        print(f"[transcripts] {video_id}: native fetch failed ({type(e).__name__})")
        return None


def fetch_transcript_whisper(video_id: str) -> str:
    """OpenAI Whisper transcription — primary path when use_whisper=True.

    Requires OPENAI_API_KEY env var, ffmpeg on PATH, and yt-dlp (hard dep).
    Raises WhisperUnavailableError on any unmet prerequisite or runtime failure.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise WhisperUnavailableError("OPENAI_API_KEY not set")

    with tempfile.TemporaryDirectory() as tmp:
        outtmpl = f"{tmp}/{video_id}.%(ext)s"
        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as e:
            raise WhisperUnavailableError(f"yt-dlp download failed: {e}")

        files = [f for f in os.listdir(tmp) if f.startswith(video_id)]
        if not files:
            raise WhisperUnavailableError("yt-dlp produced no output file")
        audio_path = os.path.join(tmp, files[0])

        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if size_mb > 25:
            raise WhisperUnavailableError(
                f"audio too large for Whisper API: {size_mb:.1f}MB (limit 25MB)"
            )

        # COST-SENSITIVE: this function makes exactly one paid OpenAI call per
        # video. If you change this — adding retries, parallelism, batching,
        # streaming, etc. — update the cost ceiling docs in PHASE_1A_SPEC.md
        # (Resolved decisions → Transcripts).
        try:
            client = OpenAI(api_key=api_key)
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    file=f, model="whisper-1"
                )
            return response.text
        except Exception as e:
            raise WhisperUnavailableError(f"Whisper API failed: {e}")


def transcribe_video(
    video_id: str, use_whisper: bool = True
) -> tuple[str | None, str]:
    """Returns (text, source). source is one of:
      'whisper'                → Whisper transcribed (use_whisper=True path)
      'youtube-transcript-api' → native fetch succeeded (use_whisper=False path)
      'unavailable'            → primary path failed; no fallback chain.
    text is None iff source == 'unavailable'.

    No cross-path fallback: if use_whisper=True and Whisper fails, the video is
    marked unavailable rather than retrying via native (and inverse for False).
    Falling between paths would surprise operators with mixed cost profiles
    and silent dispatcher behavior.
    """
    if use_whisper:
        try:
            return fetch_transcript_whisper(video_id), "whisper"
        except WhisperUnavailableError as e:
            print(f"[transcripts] {video_id}: whisper unavailable ({e})")
        except Exception as e:
            print(f"[transcripts] {video_id}: whisper failed ({type(e).__name__}: {e})")
    else:
        text = fetch_transcript_native(video_id)
        if text:
            return text, "youtube-transcript-api"

    return None, "unavailable"
