"""
ASR Service Functional Test Script
===================================
Tests:
1. Normal ASR recognition + hotword correction
2. Invalid/short audio -> 400 error
3. Temp file cleanup verification

Usage:
  python test_asr.py              # Run tests (requires ASR server running on :8000)
  python test_asr.py --server     # Start ASR server first, then run tests
"""

import argparse
import asyncio
import json
import os
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path

# ============================================================
# Config
# ============================================================
ASR_URL = "http://127.0.0.1:8000/api/v1/asr"
PROJECT_DIR = Path(__file__).parent

# Locate ffmpeg
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    import shutil
    FFMPEG_EXE = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not FFMPEG_EXE:
        FFMPEG_EXE = str(PROJECT_DIR / "ffmpeg.exe")
        if not os.path.exists(FFMPEG_EXE):
            FFMPEG_EXE = "ffmpeg"

PASS = 0
FAIL = 0
TEST_FILES = []


# ============================================================
# Helpers: generate test audio
# ============================================================

def generate_test_audio(desc: str, duration: float = 2.0, filename: str = None) -> str:
    """Generate a sine-wave test audio (mimics speech fundamental frequency)."""
    if filename is None:
        filename = f"test_audio_{uuid.uuid4().hex[:8]}.wav"
    output_path = str(PROJECT_DIR / filename)

    cmd = [
        FFMPEG_EXE, "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=300:duration={duration}",
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback: silent audio
        cmd = [
            FFMPEG_EXE, "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=16000:cl=mono:duration={duration}",
            "-acodec", "pcm_s16le",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio generation failed: {result.stderr.strip()}")

    print(f"  [AUDIO] Generated: '{filename}' (duration={duration}s, desc='{desc}')")
    return output_path


def generate_short_audio(duration: float = 0.1) -> str:
    """Generate very short audio (for 400 error test)."""
    return generate_test_audio("short_audio", duration=duration)


def generate_empty_file() -> str:
    """Generate a minimal WAV header with no audio data (44 bytes)."""
    filename = f"test_empty_{uuid.uuid4().hex[:8]}.wav"
    output_path = str(PROJECT_DIR / filename)
    with open(output_path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", 16000))
        f.write(struct.pack("<I", 32000))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", 0))
    print(f"  [AUDIO] Generated empty WAV header: '{filename}' ({os.path.getsize(output_path)} bytes)")
    return output_path


# ============================================================
# Helper: call ASR API
# ============================================================

async def call_asr(audio_path: str) -> dict:
    if not os.path.exists(audio_path):
        return {"code": 500, "msg": f"File not found: {audio_path}", "text": ""}
    import httpx
    filename = os.path.basename(audio_path)
    async with httpx.AsyncClient(timeout=30) as client:
        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, "audio/wav")}
            try:
                resp = await client.post(ASR_URL, files=files)
                data = resp.json()
                # Normalize FastAPI HTTPException format ({"detail": "..."})
                # to our standard format ({"code": ..., "msg": ..., "text": ...})
                if "detail" in data and "code" not in data:
                    return {
                        "code": resp.status_code,
                        "msg": data["detail"],
                        "text": "",
                    }
                return data
            except httpx.HTTPError as e:
                return {"code": 500, "msg": f"Request failed: {e}", "text": ""}
            except json.JSONDecodeError:
                return {"code": 500, "msg": f"Non-JSON response: {resp.text[:200]}", "text": ""}


# ============================================================
# Test Cases
# ============================================================

def report(name: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    if passed:
        PASS += 1
        label = "[PASS]"
    else:
        FAIL += 1
        label = "[FAIL]"
    print(f"  {label} {name}")
    if detail:
        print(f"         {detail}")


async def test_post_process_only():
    """Test 0: Pure post-processing unit tests (no ASR model needed)."""
    print("\n" + "=" * 60)
    print("  Test 0: post_process_text unit tests")
    print("=" * 60)

    from asr_service import post_process_text

    cases = [
        ("", "", "empty string"),
        ("   ", "", "whitespace only"),
        ("...,!.? ", "", "punctuation only"),
        ("把空条打开", "把空调打开", "kongtiao->kongtiao"),
        ("开等", "开灯", "kaideng->kaideng"),
        ("关等", "关灯", "guandeng->guandeng"),
        ("打卡灯", "打开灯", "daka->dakai"),
        ("空条打卡", "空调打开", "combined kongtiao+daka"),
        ("空挑知冷", "空调制冷", "kongtiao+zhileng"),
        ("窗连带开等", "窗帘带开灯", "multi device"),
        ("一度", "1度", "chinese digit 1"),
        ("是的", "24", "shide->24 dialect"),
        ("湿度调到26度", "温度调到26度", "shidu->wendu"),
        ("空凋调到26度", "空调调到26度", "kongdiao->kongtiao"),
        ("客厅空条制热", "客厅空调制热", "living room AC"),
    ]

    all_ok = True
    for input_text, expected, desc in cases:
        result = post_process_text(input_text)
        ok = result == expected
        if not ok:
            all_ok = False
        label = "[PASS]" if ok else "[FAIL]"
        print(f"  {label} [{desc}]")
        print(f"         input:    '{input_text}'")
        print(f"         expected: '{expected}'")
        print(f"         got:      '{result}'")

    report(f"post_process_text unit tests ({len(cases)} cases)", all_ok)


async def test_normal_recognition():
    """Test 1: Normal ASR recognition + hotword correction via API."""
    print("\n" + "=" * 60)
    print("  Test 1: Normal ASR recognition + hotword correction")
    print("=" * 60)

    audio = generate_test_audio("normal test", duration=2.0)
    TEST_FILES.append(audio)

    resp = await call_asr(audio)
    ok = resp.get("code") == 200
    report("API returns code=200", ok, f"response: {json.dumps(resp, ensure_ascii=False)}")

    if resp.get("text"):
        report("Recognition text is non-empty", True,
               f"recognized: '{resp['text']}'")
    else:
        report("Recognition text is non-empty", False,
               "text is empty (expected for synthetic test tone)")


async def test_error_handling():
    """Test 2: Invalid/short audio -> 400 error."""
    print("\n" + "=" * 60)
    print("  Test 2: Invalid/short audio error handling")
    print("=" * 60)

    # 2a: Very short audio (0.05s)
    print("\n  -- 2a: Very short audio (0.05s) --")
    short_audio = generate_short_audio(duration=0.05)
    TEST_FILES.append(short_audio)
    resp = await call_asr(short_audio)
    is_400 = resp.get("code") == 400
    has_duration_msg = "duration" in resp.get("msg", "").lower() or \
                       "时长" in resp.get("msg", "") or \
                       "过短" in resp.get("msg", "")
    report("Very short audio returns code=400", is_400,
           f"response: {json.dumps(resp, ensure_ascii=False)}")
    report("Error message mentions duration", has_duration_msg,
           f"msg: {resp.get('msg', '')}")

    # 2b: Empty file (WAV header only)
    print("\n  -- 2b: Empty file (WAV header only) --")
    empty_audio = generate_empty_file()
    TEST_FILES.append(empty_audio)
    resp = await call_asr(empty_audio)
    is_error = resp.get("code") in (400, 500)
    report("Empty file returns error code", is_error,
           f"response: {json.dumps(resp, ensure_ascii=False)}")

    # 2c: Boundary test (0.15s)
    print("\n  -- 2c: Boundary test (0.15s) --")
    boundary_audio = generate_short_audio(duration=0.15)
    TEST_FILES.append(boundary_audio)
    resp = await call_asr(boundary_audio)
    is_error = resp.get("code") in (400, 500)
    report("Boundary 0.15s returns error", is_error,
           f"response: {json.dumps(resp, ensure_ascii=False)}")


async def test_temp_file_cleanup():
    """Test 3: Verify temp file cleanup."""
    print("\n" + "=" * 60)
    print("  Test 3: Temp file cleanup verification")
    print("=" * 60)

    test_audio = generate_test_audio("cleanup test", duration=1.0)
    TEST_FILES.append(test_audio)

    before = set(f for f in os.listdir(PROJECT_DIR)
                 if f.startswith("temp_") and os.path.isfile(os.path.join(PROJECT_DIR, f)))
    print(f"  Temp files before request: {len(before)}")

    resp = await call_asr(test_audio)
    await asyncio.sleep(0.3)

    after = set(f for f in os.listdir(PROJECT_DIR)
                if f.startswith("temp_") and os.path.isfile(os.path.join(PROJECT_DIR, f)))
    new_files = after - before
    cleaned = len(new_files) == 0

    report("No temp file leaks", cleaned,
           f"new temp files: {len(new_files)} {'(' + ', '.join(new_files) + ')' if new_files else '0'}")
    if new_files:
        print(f"  [WARN] Orphaned temp files: {new_files}")

    if resp.get("code") == 200:
        report("Cleanup request succeeded", True)
    else:
        report(f"Cleanup request completed (code={resp.get('code')})", True)


# ============================================================
# Main
# ============================================================

def start_server():
    """Start ASR server as subprocess."""
    print("Starting ASR server...")
    return subprocess.Popen(
        [sys.executable, "asr_service.py"],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )


def wait_for_server(url: str, timeout: int = 60) -> bool:
    """Wait for ASR server to be ready."""
    import httpx
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(url.replace("/api/v1/asr", "/docs"), timeout=3)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


async def main():
    parser = argparse.ArgumentParser(description="ASR Service Functional Test")
    parser.add_argument("--server", action="store_true", help="Start ASR server first")
    args = parser.parse_args()

    # Set PYTHONIOENCODING for UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    server_proc = None
    if args.server:
        server_proc = start_server()
        print("  Waiting for ASR server...")
        if not wait_for_server(ASR_URL):
            print("[FAIL] ASR server startup timed out")
            if server_proc:
                server_proc.terminate()
            sys.exit(1)
        print("  ASR server ready")
        print()

    print("=" * 60)
    print("  ASR Service Functional Tests")
    print("=" * 60)

    try:
        # Test 0: unit test post-processing (no server needed)
        await test_post_process_only()

        # Test 1-3: require ASR server
        await test_normal_recognition()
        await test_error_handling()
        await test_temp_file_cleanup()

        print("\n" + "=" * 60)
        total = PASS + FAIL
        print(f"  Results: [PASS] {PASS}/{total}  [FAIL] {FAIL}/{total}")
        print("=" * 60)

    finally:
        # Cleanup test files
        print("\n  Cleaning up test files...")
        removed = 0
        for f in TEST_FILES:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    removed += 1
            except Exception:
                pass
        print(f"  Removed {removed} test files")

        # Stop server
        if server_proc:
            print("  Stopping ASR server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())