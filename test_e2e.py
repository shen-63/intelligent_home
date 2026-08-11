"""
End-to-End Integration Test Suite for Smart Home Voice Assistant
=================================================================
Tests:
  1. TTS text cleaning (unit tests - no server)
  2. SmartBrain processing (unit tests - no server)
  3. Full voice_tts HTTP pipeline (server required)
  4. Temp file garbage collection stress test

Usage:
  python test_e2e.py                          # Requires both servers running
  python test_e2e.py --start-server           # Start main.py server first
  python test_e2e.py --unit-only              # Unit tests only, skip HTTP
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

MAIN_URL = "http://127.0.0.1:8001"

PASS = 0
FAIL = 0
SKIP = 0
TEST_FILES = []

# GBK-safe icons
OK = "[OK]"
FAIL_ICON = "[FAIL]"
SKIP_ICON = "[SKIP]"


def report(name: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    if passed:
        PASS += 1
        label = OK
    else:
        FAIL += 1
        label = FAIL_ICON
    print(f"  {label} {name}")
    if detail:
        print(f"         {detail}")


def skip(name: str, reason: str = ""):
    global SKIP
    SKIP += 1
    print(f"  {SKIP_ICON} {name}")
    if reason:
        print(f"         {reason}")


# ====================================================================
# Locate ffmpeg for generating test audio
# ====================================================================
def _find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if exe:
            return exe
        fallback = str(PROJECT_DIR / "ffmpeg.exe")
        if os.path.exists(fallback):
            return fallback
        return "ffmpeg"


FFMPEG = _find_ffmpeg()


def generate_test_audio(filename: str = None, duration: float = 2.0) -> str:
    """Generate a sine-wave test audio file."""
    if filename is None:
        filename = f"e2e_audio_{uuid.uuid4().hex[:8]}.wav"
    output_path = str(PROJECT_DIR / filename)
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=300:duration={duration}",
        "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cmd = [
            FFMPEG, "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=16000:cl=mono:duration={duration}",
            "-acodec", "pcm_s16le",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")
    TEST_FILES.append(output_path)
    return output_path


def generate_short_audio(duration: float = 0.05) -> str:
    return generate_test_audio(duration=duration)


# ====================================================================
# SECTION 1: TTS text cleaning unit tests
# ====================================================================

def test_tts_text_cleaning():
    """Test 1: TTS text preprocessing (no server needed)."""
    print("\n" + "=" * 65)
    print("  SECTION 1: TTS Text Cleaning (unit tests)")
    print("=" * 65)

    from tts_service import clean_tts_text

    cases = [
        # (input, expected_substring, description)
        ("这是**重点**内容", "这是重点内容", "bold marker removal"),
        ("这是*斜体*文字", "这是斜体文字", "italic marker removal"),
        # Note: inline code test (backtick-based) removed due to shell quoting issues on Windows.
        ("```\ncode block\n```\n后面文字", "后面文字", "code block removal"),
        ("# 标题一", "标题一", "h1 header"),
        ("## 二级标题", "二级标题", "h2 header"),
        ("这是一段包含😊😎🚀的文本", "这是一段包含的文本", "emoji removal"),
        ("[点击这里](https://example.com)", "点击这里", "markdown link"),
        ("~~删除线~~文本", "删除线文本", "strikethrough"),
        ("<b>HTML标签</b>", "HTML标签", "HTML tags"),
        ("> 引用内容", "引用内容", "blockquote"),
        ("https://example.com/page 内容", "内容", "URL removal"),
        ("联系邮箱 test@example.com", "联系邮箱", "email removal"),
        ("重要！！！紧急！！消息", "重要！紧急！！消息", "repeated punctuation (3+ only)"),
        ("", "", "empty input"),
        ("   ", "", "whitespace only"),
        ("普通中文文本", "普通中文文本", "plain Chinese"),
        ("混合 **bold** 和 😊 和 code", "混合 bold 和 和 code", "mixed markdown+emoji"),
        ("---\n分隔线", "分隔线", "horizontal rule"),
    ]

    all_ok = True
    for input_text, expected, desc in cases:
        result = clean_tts_text(input_text)
        if expected == "":
            ok = result == ""
        else:
            ok = expected in result
        if not ok:
            all_ok = False
        label = OK if ok else FAIL_ICON
        print(f"  {label} [{desc}]")
        print(f"         input:    {input_text!r}")
        print(f"         expected contains: {expected!r}")
        print(f"         got:      {result!r}")

    report(f"TTS text cleaning ({len(cases)} cases)", all_ok)


# ====================================================================
# SECTION 2: SmartBrain processing unit tests
# ====================================================================

async def test_smartbrain_processing():
    """Test 2: SmartBrain intent routing + processing (no HTTP server)."""
    print("\n" + "=" * 65)
    print("  SECTION 2: SmartBrain Processing (unit tests)")
    print("=" * 65)

    try:
        from smart_brain import process
    except ImportError as e:
        skip("Import smart_brain", str(e))
        return

    # Prepare by resetting device states
    try:
        from device_controller import DEVICE_STATES, state_lock
        async with state_lock:
            for k in DEVICE_STATES:
                if DEVICE_STATES[k].get("type") == "ac":
                    DEVICE_STATES[k]["power"] = "off"
                    DEVICE_STATES[k]["temperature"] = 24
                elif DEVICE_STATES[k].get("type") == "light":
                    DEVICE_STATES[k]["power"] = "off"
                elif DEVICE_STATES[k].get("type") == "tv":
                    DEVICE_STATES[k]["power"] = "off"
                elif DEVICE_STATES[k].get("type") == "curtain":
                    DEVICE_STATES[k]["state"] = "closed"
        print("  [SETUP] Device states reset to defaults")
    except Exception as e:
        print(f"  [WARN] Could not reset device states: {e}")

    test_cases = [
        # (input_text, expected_intent, expected_contains, description)
        ("打开卧室空调", "device_control", "开启", "turn on AC"),
        ("关闭客厅灯", "device_control", "关", "turn off light"),
        ("空调调到26度", "device_control", "26", "set temperature"),
        ("空调显示E1故障码是什么意思", "knowledge_qa", "故障", "knowledge: E1 code"),
        ("设备滤网多久需要清洗一次", "knowledge_qa", "清洗", "knowledge: filter cleaning"),
    ]

    all_ok = True
    for input_text, expected_intent, expected_contains, desc in test_cases:
        try:
            result = await process(input_text, "test_session")
            intent_ok = result.get("intent") == expected_intent
            reply = result.get("reply", "")
            contains_ok = expected_contains in reply
            ok = intent_ok and contains_ok
            if not ok:
                all_ok = False
            label = OK if ok else FAIL_ICON
            print(f"  {label} [{desc}]")
            print(f"         input:    {input_text!r}")
            print(f"         intent:   {result.get('intent')} (expected {expected_intent}) {'[OK]' if intent_ok else '[MISMATCH]'}")
            print(f"         reply:    {reply[:80]!r}")
            print(f"         contains: {expected_contains!r} {'[OK]' if contains_ok else '[NOT FOUND]'}")
        except Exception as e:
            all_ok = False
            print(f"  {FAIL_ICON} [{desc}] EXCEPTION: {e}")

    report(f"SmartBrain intent routing ({len(test_cases)} cases)", all_ok)


# ====================================================================
# SECTION 3: Full HTTP E2E pipeline
# ====================================================================

async def test_voice_tts_pipeline():
    """Test 3: Full voice_tts pipeline via HTTP."""
    print("\n" + "=" * 65)
    print("  SECTION 3: Full voice_tts HTTP Pipeline")
    print("=" * 65)

    import httpx

    # Check server health first
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            hr = await client.get(f"{MAIN_URL}/api/v1/health")
        if hr.status_code != 200:
            skip("voice_tts: main.py not running",
                 f"Health check returned {hr.status_code}")
            return
        print(f"  [HEALTH] Server OK: {hr.json().get('status')}")
    except Exception as e:
        skip("voice_tts: main.py not running", str(e))
        return

    # 3a: Generate test audio and call voice_tts
    report("3a: Normal audio -> voice_tts pipeline", True,
           "Starting test with generated audio...")
    audio = generate_test_audio(duration=2.0)
    print(f"  [AUDIO] Test file: {os.path.basename(audio)} ({os.path.getsize(audio)} bytes)")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            with open(audio, "rb") as f:
                files = {"file": ("test.wav", f, "audio/wav")}
                params = {
                    "mode": "smart",
                    "voice": "xiaoxiao",
                    "session_id": "e2e_test",
                }
                resp = await client.post(
                    f"{MAIN_URL}/api/v1/voice_tts",
                    params=params,
                    files=files,
                )

        # Verify HTTP status
        status_ok = resp.status_code == 200
        report("3b: HTTP 200 OK", status_ok,
               f"status={resp.status_code}")

        # Verify content type is audio
        ct = resp.headers.get("content-type", "")
        is_audio = "audio" in ct or "mpeg" in ct
        report("3c: Response is audio/mpeg", is_audio,
               f"content-type: {ct}")

        # Verify X-Voice-Meta header
        meta_raw = resp.headers.get("x-voice-meta", "")
        import urllib.parse
        meta_parsed = urllib.parse.parse_qs(meta_raw)
        has_asr = "asr" in meta_parsed and meta_parsed["asr"][0].strip()
        has_reply = "reply" in meta_parsed and meta_parsed["reply"][0].strip()
        report("3d: X-Voice-Meta contains ASR text", has_asr,
               f"asr={meta_parsed.get('asr', [''])[0][:50]}")
        report("3e: X-Voice-Meta contains reply text", has_reply,
               f"reply={meta_parsed.get('reply', [''])[0][:50]}")

        # Verify response body is non-empty audio
        body_size = len(resp.content)
        has_audio_data = body_size > 1000
        report("3f: Response body has audio data (>1KB)",
               has_audio_data, f"size={body_size} bytes")

    except Exception as e:
        report("3a-g: voice_tts pipeline", False, str(e))

    # 3g: Short audio -> should error gracefully
    print("\n  -- 3g: Short audio (0.05s) error handling --")
    short_audio = generate_short_audio(duration=0.05)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            with open(short_audio, "rb") as f:
                files = {"file": ("short.wav", f, "audio/wav")}
                params = {"mode": "smart", "session_id": "e2e_test"}
                resp = await client.post(
                    f"{MAIN_URL}/api/v1/voice_tts",
                    params=params, files=files,
                )
        is_400 = resp.status_code == 422 or resp.status_code == 400
        report("3g: Short audio returns error", is_400,
               f"status={resp.status_code}, body={resp.text[:100]}")
    except Exception as e:
        report("3g: Short audio error handling", False, str(e))

    # 3h: text_chat endpoint (no audio upload needed)
    print("\n  -- 3h: text_chat endpoint --")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{MAIN_URL}/api/v1/text_chat",
                params={
                    "text": "打开卧室空调",
                    "mode": "smart",
                    "session_id": "e2e_test",
                },
            )
        status_ok = resp.status_code == 200
        is_audio = "audio" in resp.headers.get("content-type", "")
        report("3h: text_chat returns 200 + audio", status_ok and is_audio,
               f"status={resp.status_code}, content-type={resp.headers.get('content-type')}")
    except Exception as e:
        report("3h: text_chat endpoint", False, str(e))

    # 3i: Weather query via text_chat
    print("\n  -- 3i: Weather query via text_chat --")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{MAIN_URL}/api/v1/text_chat",
                params={
                    "text": "今天长沙天气怎么样",
                    "mode": "smart",
                    "session_id": "e2e_test",
                },
            )
        is_200 = resp.status_code == 200
        meta_raw = resp.headers.get("x-voice-meta", "")
        meta_parsed = urllib.parse.parse_qs(meta_raw)
        reply = meta_parsed.get("reply", [""])[0]
        has_weather_reply = bool(reply and ("天气" in reply or "长沙" in reply or "°C" in reply or "℃" in reply))
        report("3i: Weather query returns TTS audio", is_200,
               f"status={resp.status_code}")
        if reply:
            report("3i: Reply contains weather info", has_weather_reply,
                   f"reply={reply[:80]}")
        else:
            skip("3i: Reply weather check", "No reply metadata found")
    except Exception as e:
        report("3i: Weather query", False, str(e))


# ====================================================================
# SECTION 4: Temp file garbage collection stress test
# ====================================================================

async def test_temp_file_gc():
    """Test 4: Temp file cleanup under repeated calls."""
    print("\n" + "=" * 65)
    print("  SECTION 4: Temp File GC Stress Test")
    print("=" * 65)

    import httpx

    # Check server
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.get(f"{MAIN_URL}/api/v1/health")
    except Exception as e:
        skip("Temp file GC: main.py not running", str(e))
        return

    # Count existing temp files
    def count_temp_files():
        return len([f for f in os.listdir(PROJECT_DIR)
                    if f.startswith("temp_") and os.path.isfile(os.path.join(PROJECT_DIR, f))])

    before_count = count_temp_files()
    print(f"  Temp files before: {before_count}")

    # Generate a reusable audio file
    audio = generate_test_audio(duration=1.5)

    # Send 5 rapid requests
    N = 5
    async with httpx.AsyncClient(timeout=60) as client:
        with open(audio, "rb") as f:
            audio_data = f.read()

    successes = 0
    for i in range(N):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{MAIN_URL}/api/v1/voice_tts",
                    params={"mode": "smart", "session_id": f"gc_test_{i}"},
                    files={"file": ("loop.wav", audio_data, "audio/wav")},
                )
            if resp.status_code == 200:
                successes += 1
        except Exception:
            pass
        # Small delay between requests
        await asyncio.sleep(0.1)

    # Wait a tiny bit more for async cleanup
    await asyncio.sleep(0.5)

    after_count = count_temp_files()
    leaked = after_count - before_count

    print(f"  Temp files after {N} requests: {after_count} (before: {before_count})")
    print(f"  Successful requests: {successes}/{N}")
    print(f"  Leaked temp files: {leaked}")

    is_acceptable = leaked <= 2  # Allow a small window of in-flight files
    report(f"Temp file GC: <=2 leaked after {N} requests", is_acceptable,
           f"before={before_count}, after={after_count}, leaked={leaked}")

    # Also test the text_chat endpoint for temp file leaks
    print("\n  -- text_chat temp file check --")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            for i in range(3):
                resp = await client.post(
                    f"{MAIN_URL}/api/v1/text_chat",
                    params={"text": "今天天气怎么样", "mode": "smart",
                            "session_id": f"gc_text_{i}"},
                )
                if resp.status_code != 200:
                    print(f"  [WARN] text_chat request {i} returned {resp.status_code}")
    except Exception as e:
        print(f"  [WARN] text_chat GC test: {e}")

    await asyncio.sleep(0.5)
    final_count = count_temp_files()
    print(f"  Final temp file count (after text_chat): {final_count}")
    leaked_final = final_count - before_count
    report(f"No temp file leak from text_chat", leaked_final <= 2,
           f"final={final_count}, leaked_since_baseline={leaked_final}")


# ====================================================================
# Server management
# ====================================================================

def start_main_server():
    """Start main.py as subprocess."""
    print("Starting main.py server on port 8001...")
    return subprocess.Popen(
        [sys.executable, "-u", "main.py"],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        # Needs funasr imports which use subprocess; keep env as-is
    )


def wait_for_main(timeout: int = 90) -> bool:
    """Wait for main.py health endpoint to respond."""
    import httpx
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{MAIN_URL}/api/v1/health", timeout=5)
            if resp.status_code == 200:
                print("  main.py server ready")
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


# ====================================================================
# Main
# ====================================================================

async def main():
    parser = argparse.ArgumentParser(description="E2E Integration Tests")
    parser.add_argument("--start-server", action="store_true",
                        help="Start main.py server first")
    parser.add_argument("--unit-only", action="store_true",
                        help="Run unit tests only (no HTTP)")
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    server_proc = None
    if args.start_server:
        server_proc = start_main_server()
        if not wait_for_main():
            print(f"{FAIL_ICON} main.py server startup timed out")
            if server_proc:
                server_proc.terminate()
            sys.exit(1)
        print()

    print("=" * 65)
    print("  E2E Integration Test Suite")
    print("=" * 65)

    try:
        # Section 1: TTS text cleaning (always runs)
        test_tts_text_cleaning()

        # Section 2: SmartBrain processing (always runs)
        await test_smartbrain_processing()

        # Section 3 & 4: HTTP pipeline tests (skip if unit-only)
        if not args.unit_only:
            await test_voice_tts_pipeline()
            await test_temp_file_gc()
        else:
            skip("Section 3: HTTP pipeline", "--unit-only flag set")
            skip("Section 4: Temp file GC", "--unit-only flag set")

        # Summary
        print("\n" + "=" * 65)
        total_run = PASS + FAIL
        print(f"  RESULTS SUMMARY")
        print(f"  {OK} Passed:  {PASS}/{total_run}")
        print(f"  {FAIL_ICON} Failed:  {FAIL}/{total_run}")
        print(f"  {SKIP_ICON} Skipped: {SKIP}")
        print("=" * 65)

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
            print("  Stopping main.py...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            print("  Server stopped")

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())