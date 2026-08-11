"""TTS 语音合成服务

基于 edge-tts (Microsoft Edge 在线 TTS) 的轻量封装。
提供文本预处理、文件/字节/响应三种合成模式，以及临时文件清理。
"""

import asyncio
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import edge_tts

# ============ 默认配置 ============

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_RATE = "+0%"
DEFAULT_VOLUME = "+0%"

# 中文语音别名 → edge-tts voice id
CHINESE_VOICES = {
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",
    "yunxi": "zh-CN-YunxiNeural",
    "xiaoyi": "zh-CN-XiaoyiNeural",
    "yunyang": "zh-CN-YunyangNeural",
    "xiaohui": "zh-CN-XiaohuiNeural",
}

# 临时 TTS 文件保留时长（秒）
TEMP_TTL_SEC = 300


# ============ 文本预处理 ============

def clean_tts_text(raw_text: str) -> str:
    """TTS-specific text preprocessing pipeline.

    Removes content that would produce garbled/misleading speech output:
    1. Markdown formatting (bold, italic, headers, inline code, code blocks, links)
    2. Emoji and emoticons
    3. HTML tags
    4. Repeated/unusual punctuation sequences
    5. URL/email patterns

    Args:
        raw_text: Original text (may contain markdown, emoji, etc.)

    Returns:
        Cleaned text suitable for TTS synthesis
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text

    # 代码块 ```...```
    text = re.sub(r"```[^`]*```", "", text)
    # 行内代码 `...`
    text = re.sub(r"`[^`]+`", "", text)

    # 链接 [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # 图片 ![alt](url) → 空
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)

    # 标题 # → 空
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # 粗体/斜体 *text* / **text** / ***text***
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)

    # 删除线 ~~text~~ → text
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # 水平分割线 --- / *** / ___
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # 引用 > text → text
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    # HTML 标签
    text = re.sub(r"<[^>]+>", "", text)

    # Emoji 与符号表情（覆盖常见 Unicode emoji 区段）
    text = re.sub(
        "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff"
        "\u2702-\u27b0\U0001f850-\U0001f8ff\U0001fa00-\U0001fa6f"
        "\U0001fa70-\U0001faff\ufe00-\ufe0f\u2600-\u26ff]+",
        "",
        text,
    )

    # URL / 邮箱
    text = re.sub(r"https?://[^\s,，。！？；;]+", "", text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.+-]+", "", text)

    # 重复标点收敛
    text = re.sub(r"([。！？，、；：]){3,}", r"\1", text)
    text = re.sub(r"([.!?,;:]){3,}", r"\1", text)
    text = re.sub(r"。{2,}", "。", text)
    text = re.sub(r"\.{3,}", "。", text)

    # 空白合并
    text = re.sub(r"\s+", " ", text).strip()
    # 去掉首尾残留标点
    text = text.strip("，。！？、；：,.!?;:'\" ")
    return text


# ============ 临时文件清理 ============

def _cleanup_stale_temp_files(tts_dir: str, max_age: int = TEMP_TTL_SEC) -> None:
    """清理 tts_dir 中超过 max_age 秒的 tts_*.mp3 临时文件。"""
    if not os.path.isdir(tts_dir):
        return
    now = time.time()
    for fname in os.listdir(tts_dir):
        if not fname.startswith("tts_"):
            continue
        fpath = os.path.join(tts_dir, fname)
        try:
            if now - os.path.getmtime(fpath) > max_age:
                os.remove(fpath)
        except (OSError, PermissionError):
            continue


# ============ 合成接口 ============

async def synthesize(
    text: str,
    output_path: str,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    volume: str = DEFAULT_VOLUME,
) -> str:
    """Text-to-Speech synthesis, save result to file.

    Automatically preprocesses text (removes markdown/emoji) before synthesis.
    Runs edge-tts which uses httpx internally (non-blocking).

    Args:
        text: Raw text to synthesize (will be cleaned)
        output_path: Output audio file path (.mp3)
        voice: Voice name
        rate: Speed adjustment (+0%, -50%, +100%)
        volume: Volume adjustment

    Returns:
        Output file path
    """
    cleaned = clean_tts_text(text)
    if not cleaned:
        raise ValueError("Text is empty after TTS preprocessing")

    parent_dir = os.path.dirname(output_path) or "."
    _cleanup_stale_temp_files(parent_dir)

    communicate = edge_tts.Communicate(cleaned, voice, rate=rate, volume=volume)
    await communicate.save(output_path)
    return output_path


async def synthesize_to_bytes(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    volume: str = DEFAULT_VOLUME,
) -> bytes:
    """Text-to-Speech synthesis, return audio bytes.

    Uses a temp file internally; cleaned up automatically.

    Args:
        text: Raw text to synthesize (will be cleaned)
        voice: Voice name
        rate: Speed adjustment
        volume: Volume adjustment

    Returns:
        MP3 audio bytes
    """
    cleaned = clean_tts_text(text)
    if not cleaned:
        raise ValueError("Text is empty after TTS preprocessing")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name

    try:
        communicate = edge_tts.Communicate(cleaned, voice, rate=rate, volume=volume)
        await communicate.save(temp_path)
        with open(temp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


async def synthesize_to_response(
    text: str,
    tts_dir: str = ".",
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    volume: str = DEFAULT_VOLUME,
) -> dict:
    """One-shot TTS: clean text, synthesize, return file + metadata.

    This is the preferred helper for endpoint handlers — it manages the
    full lifecycle (clean -> synthesize -> cleanup stale -> return path).

    Args:
        text: Raw text
        tts_dir: Directory to store the temp audio file
        voice: Voice name
        rate: Speed adjustment
        volume: Volume adjustment

    Returns:
        {
            "file_path": str,       # Path to the generated MP3
            "cleaned_text": str,    # Text after TTS preprocessing
        }
    """
    cleaned = clean_tts_text(text)
    if not cleaned:
        raise ValueError("Text is empty after TTS preprocessing")

    os.makedirs(tts_dir, exist_ok=True)
    _cleanup_stale_temp_files(tts_dir)

    task_id = str(uuid.uuid4().hex[:12])
    output_path = os.path.join(tts_dir, f"tts_{task_id}.mp3")

    communicate = edge_tts.Communicate(cleaned, voice, rate=rate, volume=volume)
    await communicate.save(output_path)

    return {
        "file_path": output_path,
        "cleaned_text": cleaned,
    }


# ============ 语音列表 ============

def list_voices() -> list:
    """Get all available Chinese voices as a list of dicts."""
    return [{"name": k, "voice": v} for k, v in CHINESE_VOICES.items()]
