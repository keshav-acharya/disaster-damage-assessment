"""
API test suite for the AI Disaster Damage Assessment backend.

Run with:
    cd disaster_assessment
    python tests/test_api.py

The FastAPI server must be running on http://localhost:8000 before running tests.
Start it with: uvicorn app.main:app --reload
"""

import io
import os
import sys
import time

import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000"
TIMEOUT = 30  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_png(width: int = 256, height: int = 256) -> bytes:
    """Create a minimal valid PNG image in memory (no file I/O needed)."""
    try:
        from PIL import Image
        img = Image.fromarray(
            np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        import struct
        import zlib

        def png_chunk(chunk_type, data):
            c = chunk_type + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        raw = b"\x00" + b"\xff" * (width * 3)
        idat_data = zlib.compress(raw * height)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + png_chunk(b"IDAT", idat_data)
            + png_chunk(b"IEND", b"")
        )
        return png


def _pass(test_name: str, msg: str = ""):
    print(f"  ✅ PASS  {test_name}" + (f" – {msg}" if msg else ""))


def _fail(test_name: str, msg: str = ""):
    print(f"  ❌ FAIL  {test_name}" + (f" – {msg}" if msg else ""))


def _section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_health_check():
    """GET /health should return 200 with status=ok."""
    _section("Test 1 – Health Check")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        body = resp.json()
        assert body.get("status") == "ok", f"Expected status=ok, got {body.get('status')}"
        _pass("GET /health", f"status=ok, mock_mode={body.get('mock_mode')}")
        print(f"       mock_mode_message: {body.get('mock_mode_message')}")
    except requests.ConnectionError:
        _fail("GET /health", "Cannot connect – is the server running on localhost:8000?")
    except AssertionError as exc:
        _fail("GET /health", str(exc))


def test_predict_valid_images():
    """POST /predict with valid PNG images should return structured JSON."""
    _section("Test 2 – Predict with Valid Images")
    png = _make_dummy_png()
    files = {
        "before_image": ("before.png", io.BytesIO(png), "image/png"),
        "after_image": ("after.png", io.BytesIO(png), "image/png"),
    }
    try:
        resp = requests.post(f"{BASE_URL}/predict", files=files, timeout=TIMEOUT)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}\n{resp.text}"
        body = resp.json()

        assert body.get("status") == "success", f"status={body.get('status')}"
        assert "total_buildings" in body, "Missing 'total_buildings'"
        assert "buildings" in body, "Missing 'buildings'"
        assert "summary" in body, "Missing 'summary'"
        assert isinstance(body["buildings"], list), "'buildings' is not a list"

        _pass("POST /predict", f"total_buildings={body['total_buildings']}, mock_mode={body.get('mock_mode')}")
        print(f"       summary: {body.get('summary')}")
        print(f"       processing_time_ms: {body.get('processing_time_ms')}")

        if body.get("mock_mode"):
            print(f"       mock_warning: {body.get('mock_warning')}")

        # Validate first building structure
        if body["buildings"]:
            b = body["buildings"][0]
            assert "bounding_box" in b, "Building missing 'bounding_box'"
            assert "damage_class" in b, "Building missing 'damage_class'"
            assert "damage_severity" in b, "Building missing 'damage_severity'"
            assert "detection_confidence" in b, "Building missing 'detection_confidence'"
            _pass("Building structure", f"damage={b['damage_class']}, conf={b['detection_confidence']}")

    except requests.ConnectionError:
        _fail("POST /predict", "Cannot connect – is the server running?")
    except AssertionError as exc:
        _fail("POST /predict", str(exc))


def test_predict_invalid_image():
    """POST /predict with a corrupted image should return 422."""
    _section("Test 3 – Predict with Invalid (Corrupted) Image")
    junk = b"this is not an image file at all"
    files = {
        "before_image": ("before.jpg", io.BytesIO(junk), "image/jpeg"),
        "after_image": ("after.png", io.BytesIO(_make_dummy_png()), "image/png"),
    }
    try:
        resp = requests.post(f"{BASE_URL}/predict", files=files, timeout=TIMEOUT)
        if resp.status_code in (400, 422):
            _pass("Invalid image rejected", f"HTTP {resp.status_code}")
        else:
            _fail("Invalid image rejected", f"Expected 422, got {resp.status_code}")
    except requests.ConnectionError:
        _fail("Invalid image rejected", "Cannot connect to server.")


def test_predict_unsupported_format():
    """POST /predict with an unsupported file extension should return 422."""
    _section("Test 4 – Predict with Unsupported Format (.bmp)")
    png = _make_dummy_png()
    files = {
        "before_image": ("before.bmp", io.BytesIO(png), "image/bmp"),
        "after_image": ("after.png", io.BytesIO(png), "image/png"),
    }
    try:
        resp = requests.post(f"{BASE_URL}/predict", files=files, timeout=TIMEOUT)
        if resp.status_code in (400, 422):
            _pass("Unsupported format rejected", f"HTTP {resp.status_code}")
            print(f"       detail: {resp.json().get('detail', '')[:100]}")
        else:
            _fail("Unsupported format rejected", f"Expected 422, got {resp.status_code}")
    except requests.ConnectionError:
        _fail("Unsupported format rejected", "Cannot connect to server.")


def test_docs_accessible():
    """GET /docs should be accessible (Swagger UI)."""
    _section("Test 5 – Swagger Docs Page")
    try:
        resp = requests.get(f"{BASE_URL}/docs", timeout=TIMEOUT)
        if resp.status_code == 200:
            _pass("GET /docs", "Swagger UI is accessible")
        else:
            _fail("GET /docs", f"HTTP {resp.status_code}")
    except requests.ConnectionError:
        _fail("GET /docs", "Cannot connect to server.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AI Disaster Damage Assessment – API Test Suite")
    print(f"  Target: {BASE_URL}")
    print("=" * 60)

    test_health_check()
    test_predict_valid_images()
    test_predict_invalid_image()
    test_predict_unsupported_format()
    test_docs_accessible()

    print("\n" + "=" * 60)
    print("  Test run complete.")
    print("=" * 60 + "\n")
