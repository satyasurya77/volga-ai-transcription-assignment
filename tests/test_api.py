"""Offline tests: no model download required."""
import io
import wave
from types import SimpleNamespace

from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def make_wav(seconds=1):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000 * seconds)
    return stream.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_unsupported_extension():
    response = client.post("/transcribe", files={"file": ("bad.txt", b"hi", "text/plain")})
    assert response.status_code == 400


def test_empty_upload():
    response = client.post("/transcribe", files={"file": ("empty.wav", b"", "audio/wav")})
    assert response.status_code == 400


def test_oversize_upload(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    response = client.post("/transcribe", files={"file": ("big.wav", b"0" * 20, "audio/wav")})
    assert response.status_code == 413


def test_short_audio_pipeline(monkeypatch):
    monkeypatch.setattr(main, "probe_audio", lambda path: 1.0)
    monkeypatch.setattr(main, "extract_chunk", lambda *args: None)

    class FakeModel:
        def transcribe(self, path, **kwargs):
            return iter([SimpleNamespace(start=0.1, end=0.8, text=" hello world ")]), SimpleNamespace(language="en")

    monkeypatch.setattr(main, "get_model", lambda: FakeModel())
    response = client.post("/transcribe", files={"file": ("sample.wav", make_wav(), "audio/wav")})
    assert response.status_code == 200
    result = response.json()
    assert result["text"] == "hello world"
    assert result["language"] == "en"
    assert result["segments"] == [{"start": 0.1, "end": 0.8, "text": "hello world"}]


def test_long_audio_returns_job(monkeypatch):
    monkeypatch.setattr(main, "probe_audio", lambda path: 130.0)
    monkeypatch.setattr(main, "transcribe_file", lambda path, filename, duration: {"text": "done"})
    response = client.post("/transcribe", files={"file": ("long.wav", make_wav(), "audio/wav")})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    status = client.get(f"/jobs/{job_id}")
    assert status.json()["status"] == "completed"


def test_chunk_timestamp_offset(monkeypatch):
    monkeypatch.setattr(main, "extract_chunk", lambda *args: None)
    calls = iter([
        (iter([SimpleNamespace(start=1.0, end=2.0, text="first")]), SimpleNamespace(language="en")),
        (iter([SimpleNamespace(start=2.0, end=3.0, text="second")]), SimpleNamespace(language="en")),
    ])

    class FakeModel:
        def transcribe(self, path, **kwargs):
            return next(calls)

    monkeypatch.setattr(main, "get_model", lambda: FakeModel())
    result = main.transcribe_file(main.Path("ignored.wav"), "ignored.wav", 90)
    assert result["segments"][0]["start"] == 1.0
    # second chunk begins at original time 59 sec
    assert result["segments"][1]["start"] == 61.0


def test_unknown_job():
    assert client.get("/jobs/nonexistent-id").status_code == 404
