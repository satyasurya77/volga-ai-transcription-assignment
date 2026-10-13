"""CPU-only speech-to-text API using Faster-Whisper and FastAPI."""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="Audio Transcription API", version="1.0.0")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_SECONDS = 2 * 60 * 60
LONG_AUDIO_THRESHOLD_SECONDS = 120
CHUNK_SECONDS = 60
OVERLAP_SECONDS = 1
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

# Prototype-only job state: not persistent between server restarts.
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
inference_lock = threading.Lock()


class Segment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    filename: str
    language: str | None
    duration: float
    text: str
    segments: list[Segment]


@lru_cache(maxsize=1)
def get_model():
    from faster_whisper import WhisperModel

    logger.info("Loading Faster-Whisper model: %s", MODEL_NAME)
    return WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=180)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg and FFprobe must be installed and on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Audio processing timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError("Audio file cannot be decoded or has an unsupported codec") from exc


def probe_audio(path: Path) -> float:
    import json

    output = run_command([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name:format=duration",
        "-of", "json", str(path),
    ])
    try:
        metadata = json.loads(output.stdout)
        if not metadata.get("streams"):
            raise ValueError("No audio stream found")
        duration = float(metadata["format"]["duration"])
        if not (0 < duration <= MAX_DURATION_SECONDS):
            raise ValueError("Audio duration must be between 0 and 7200 seconds")
        return duration
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid audio metadata: {exc}") from exc


def extract_chunk(source: Path, destination: Path, start: float, duration: float) -> None:
    """Decode a requested section to 16 kHz mono 16-bit PCM WAV."""
    run_command([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-i", str(source), "-t", str(duration),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
    ])


def transcribe_file(path: Path, filename: str, duration: float) -> dict[str, Any]:
    """Chunked inference with absolute timestamps and heuristic overlap removal."""
    model = get_model()
    all_segments: list[dict[str, Any]] = []
    language: str | None = None
    position = 0.0

    with inference_lock:
        with tempfile.TemporaryDirectory(prefix="stt_chunks_") as folder:
            while position < duration:
                start = max(0.0, position - (OVERLAP_SECONDS if position else 0.0))
                end = min(
                    duration,
                    position + CHUNK_SECONDS
                    + (OVERLAP_SECONDS if position + CHUNK_SECONDS < duration else 0.0),
                )
                chunk_path = Path(folder) / "chunk.wav"
                extract_chunk(path, chunk_path, start, end - start)

                segments, info = model.transcribe(
                    str(chunk_path), beam_size=5, vad_filter=True
                )
                if language is None:
                    language = info.language

                for segment in segments:
                    absolute_start = max(0.0, start + float(segment.start))
                    absolute_end = min(duration, start + float(segment.end))
                    midpoint = (absolute_start + absolute_end) / 2
                    owns_segment = (
                        position <= midpoint < min(duration, position + CHUNK_SECONDS)
                        or (position + CHUNK_SECONDS >= duration and midpoint == duration)
                    )
                    content = segment.text.strip()
                    if owns_segment and content and absolute_end >= absolute_start:
                        all_segments.append({
                            "start": round(absolute_start, 2),
                            "end": round(absolute_end, 2),
                            "text": content,
                        })
                position += CHUNK_SECONDS

    all_segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    return Transcript(
        filename=filename,
        language=language,
        duration=round(duration, 2),
        text=" ".join(segment["text"] for segment in all_segments),
        segments=[Segment(**segment) for segment in all_segments],
    ).model_dump()


def process_job(job_id: str, path: Path, filename: str, duration: float) -> None:
    with jobs_lock:
        jobs[job_id] = {"status": "processing"}
    try:
        result = transcribe_file(path, filename, duration)
        with jobs_lock:
            jobs[job_id] = {"status": "completed", "result": result}
    except Exception:
        logger.exception("Transcription job failed: %s", job_id)
        with jobs_lock:
            jobs[job_id] = {"status": "failed", "error": "Transcription failed"}
    finally:
        path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/transcribe",
    response_model=Transcript,
    responses={202: {"description": "Long audio queued"}},
)
def transcribe(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported extension")

    fd, filename = tempfile.mkstemp(prefix="stt_upload_", suffix=extension)
    os.close(fd)
    path = Path(filename)
    retain_for_job = False
    try:
        total = 0
        with path.open("wb") as destination:
            while data := file.file.read(1024 * 1024):
                total += len(data)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Maximum upload size is 25 MB")
                destination.write(data)
        if total == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            duration = probe_audio(path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if duration > LONG_AUDIO_THRESHOLD_SECONDS:
            job_id = uuid.uuid4().hex
            with jobs_lock:
                jobs[job_id] = {"status": "queued"}
            background_tasks.add_task(process_job, job_id, path, original_name, duration)
            retain_for_job = True
            return JSONResponse(status_code=202, content={
                "job_id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"
            })

        return transcribe_file(path, original_name, duration)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Transcription request failed")
        raise HTTPException(status_code=500, detail="Transcription failed") from None
    finally:
        file.file.close()
        if not retain_for_job:
            path.unlink(missing_ok=True)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job ID")
        return dict(job)
