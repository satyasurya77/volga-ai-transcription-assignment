# Audio Transcription Pipeline — FastAPI

CPU-only FastAPI service for WAV, MP3, FLAC, M4A and OGG speech-to-text, with segment timestamps and long-recording support. Built for the Volga Partners AI Engineer take-home assignment.

## Prerequisites (Windows)

- Python 3.11 recommended (Python 3.10–3.12 supported)
- FFmpeg **and** FFprobe installed and on your `PATH`
- Internet access on first transcription to download Faster-Whisper model weights
- No GPU required

Verify installation in a fresh PowerShell session:

```powershell
ffmpeg -version
ffprobe -version
```

## Install

From the cloned repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `.venv\Scripts\python.exe` directly. If Python 3.11 is unavailable, use a supported installed version.

## Start server

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to try `POST /transcribe`. Alternatively:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/transcribe" -F "file=@sample_audio.wav"
```

The default model is `base`, with `device="cpu"` and `compute_type="int8"`. For slower laptops, set `$env:WHISPER_MODEL="tiny"` before starting the server. The model loads on the first transcription and is cached per process.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Health check |
| POST | `/transcribe` | Multipart upload field `file` |
| GET | `/jobs/{job_id}` | Retrieve background job status/result |

For recordings up to 120 seconds, `POST /transcribe` returns HTTP 200 with a transcript. Longer recordings return HTTP 202 with a `job_id` and relative `status_url`. Job states: queued, processing, completed, failed.

Illustrative response (not an actual measured transcription):

```json
{
  "filename": "sample_audio.wav",
  "language": "en",
  "duration": 3.5,
  "text": "Hello, welcome.",
  "segments": [
    {"start": 0.4, "end": 2.1, "text": "Hello, welcome."}
  ]
}
```

## Assignment requirements

| Requirement | Code |
| --- | --- |
| Accept audio upload | `transcribe()` using FastAPI `UploadFile`, upload validation and cleanup |
| Transcribe speech | `get_model()` and `transcribe_file()` using Faster-Whisper |
| Segment timestamps | `transcribe_file()` generates start/end/text and complete transcript |
| Different formats | `probe_audio()` validates stream; `extract_chunk()` decodes to 16 kHz mono PCM |
| Long audio | 60-second logical windows with one-second overlap, 202 responses and job polling |

## Tests

```powershell
python -m pytest -q
```

The automated tests mock the STT model and FFmpeg operations. They do **not** prove live recognition works. For end-to-end validation, upload a real WAV or MP3 after installing FFmpeg and downloading model weights.

## Design decisions and limitations

- **Validation:** allowlisted filename extensions, FFprobe stream inspection, 25 MiB upload limit, and two-hour duration limit. Invalid audio metadata is rejected.
- **Decoding:** FFmpeg decodes requested sections into mono 16 kHz PCM WAV, so the model uses consistent input even when original formats differ.
- **Chunking:** 60-second ownership windows with a one-second overlap. Absolute timestamps are rebuilt from chunk start times. Midpoint ownership heuristically reduces duplicate segments; it can lose or duplicate words near boundaries. Silence-aware segmentation/alignment would be better in production.
- **Inference:** INT8 CPU Whisper is practical for laptop evaluation. A process-local lock serializes model inference to limit contention.
- **Async jobs:** FastAPI BackgroundTasks runs in the same API process. Job state is stored only in a Python dictionary; it does not survive restarts or scale across workers. Production would use Celery/RQ, Redis, durable result storage, a bounded queue, and separate inference workers.
- **Storage:** original audio is removed from temporary storage after processing. A deployed service should use encrypted object storage, restricted access and retention policies for sensitive uploads.
- **Retry:** the prototype reports failed jobs but does not automatically retry them. Production retries should distinguish transient worker faults from corrupt inputs and implement backoff, idempotency and chunk checkpoints.
- **API security:** production needs authentication, rate limiting, observability and a controlled ingestion/storage layer.
- **Accuracy:** speaker diarization and word-level alignment are out of scope. Measure WER, latency and timestamp quality against representative audio before claiming production readiness.

## Submission status

Source code and offline unit tests are included. Real-audio transcription should be verified on the target Windows laptop before the walkthrough video.
