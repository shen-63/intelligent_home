"""
SenseVoice ASR Service for Smart Home
======================================

增强版语音识别服务，在原有基础上增加了：
1. 音频格式兼容与统一转码（使用 ffmpeg）
2. 音频有效性校验（文件大小、时长检测）
3. 智能家居热词纠错与文本清洗
4. 健壮的异常处理与临时文件清理
"""

import os
import re
import uuid
import subprocess
import imageio_ffmpeg
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from funasr import AutoModel

# ---------------------------------------------------------------------------
# 1. 动态将 imageio-ffmpeg 的路径加入系统环境变量 PATH 中
# ---------------------------------------------------------------------------
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
os.environ["PATH"] += os.pathsep + ffmpeg_dir

app = FastAPI(title="SenseVoice ASR Service for Smart Home")

# 2. 初始化加载 SenseVoice-Small 模型
print("正在加载 SenseVoice-Small 模型...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    disable_update=True,
)
print("模型加载完成！")

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 允许的音频扩展名（前端可能上传的各种格式）
ALLOWED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
    ".webm", ".pcm", ".amr", ".opus", ".wma", ".aiff",
    ".au", ".snd", ".ra", ".mp4", ".mov", ".avi",
}

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# 最小有效时长（秒）：短于该值的音频直接视为无效
MIN_AUDIO_DURATION_SEC = 0.2

# 最小有效文件大小（字节）：小于该值的文件大概率是空/损坏
MIN_FILE_SIZE = 44  # 即使是最小的有效 WAV 文件头也有 44 字节

# ---------------------------------------------------------------------------
# 智能家居热词纠错字典
# ---------------------------------------------------------------------------
# 键 = 错误的同音/近似音文本，值 = 正确文本
# 按 key 长度降序排列，优先匹配长词，避免短词误替换
HOTWORD_CORRECTIONS = [
    # ---- 灯 ----
    ("开等", "开灯"),
    ("开登", "开灯"),
    ("关等", "关灯"),
    ("关登", "关灯"),
    ("等", "灯"),       # 只在无更优匹配时生效
    # ---- 空调 ----
    ("空条", "空调"),
    ("空挑", "空调"),
    ("空凋", "空调"),
    ("控条", "空调"),
    # ---- 窗帘 ----
    ("窗连", "窗帘"),
    ("窗廉", "窗帘"),
    ("创联", "窗帘"),
    # ---- 电视 ----
    ("电事", "电视"),
    ("电示", "电视"),
    ("店视", "电视"),
    # ---- 打开/关闭 ----
    ("打卡", "打开"),
    ("大卡", "打开"),
    ("达卡", "打开"),
    ("大看", "打开"),
    ("关必", "关闭"),
    ("关逼", "关闭"),
    # ---- 温度/模式 ----
    ("知冷", "制冷"),
    ("知热", "制热"),
    ("自冷", "制冷"),
    ("自热", "制热"),
    ("质问", "制冷"),
    # ---- 常用设备名 ----
    ("卧室", "卧室"),
    ("客厅", "客厅"),
    # ---- 其他 ----
    ("一度", "1度"),
    ("二度", "2度"),
    ("三度", "3度"),
    ("四度", "4度"),
    ("五度", "5度"),
    ("湿度", "温度"),   # "空调湿度" → "空调温度"（常见误说）
    ("是的", "24"),     # 部分方言 "24" 被听成 "是的"
    ("设置", "设为"),
]

# 中文标点符号集合（用于清理）
CJK_PUNCTUATION = "。，！？、；：""''（）《》【】"


def convert_to_16k_wav(input_path: str, output_path: str) -> float:
    """
    使用 ffmpeg 强制转码为 16kHz, 单声道, 16bit PCM WAV。

    Args:
        input_path:  输入音频文件路径
        output_path: 输出 WAV 文件路径

    Returns:
        音频时长（秒），通过 ffprobe 解析获得

    Raises:
        RuntimeError: 转码失败或输出为空
    """
    # 先获取原始音频时长（用于后续校验）
    duration = _probe_duration(input_path)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", input_path,
        "-ar", "16000",        # 采样率 16kHz
        "-ac", "1",            # 单声道
        "-acodec", "pcm_s16le",  # 16bit PCM
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 转码失败: {result.stderr.strip()}")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("转码后文件为空")

    return duration


def _probe_duration(file_path: str) -> float:
    """
    使用 ffprobe 快速获取音频时长（秒）。

    若 ffprobe 不可用或解析失败，返回 0.0（不阻断流程）。
    """
    # ffprobe 通常与 ffmpeg 在同一目录
    ffprobe_candidates = [
        os.path.join(ffmpeg_dir, "ffprobe.exe"),
        os.path.join(ffmpeg_dir, "ffprobe"),
        "ffprobe",
    ]

    ffprobe_exe = None
    for candidate in ffprobe_candidates:
        if candidate.startswith("ffprobe") and candidate != "ffprobe":
            # 绝对路径检查
            if os.path.exists(candidate):
                ffprobe_exe = candidate
                break
        else:
            # 系统 PATH 检查
            try:
                subprocess.run(
                    [candidate, "-version"],
                    capture_output=True, text=True, timeout=5
                )
                ffprobe_exe = candidate
                break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

    if not ffprobe_exe:
        # 降级：使用 ffmpeg 自身打印时长信息（不如 ffprobe 精确但够用）
        return _probe_duration_via_ffmpeg(file_path)

    try:
        result = subprocess.run(
            [
                ffprobe_exe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, OSError):
        pass

    return 0.0


def _probe_duration_via_ffmpeg(file_path: str) -> float:
    """
    降级方案：使用 ffmpeg 的stderr输出解析时长。

    ffmpeg 在处理文件时会输出 "Duration: 00:01:23.45" 格式的时长行。
    """
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-i", file_path],
            capture_output=True, text=True, timeout=15,
        )
        # 时长信息在 stderr 中
        match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)",
            result.stderr,
        )
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            centiseconds = int(match.group(4))
            return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0
    except (subprocess.TimeoutExpired, OSError):
        pass

    return 0.0


# ---------------------------------------------------------------------------
# 文本后处理：热词纠错 + 标点清洗
# ---------------------------------------------------------------------------

def post_process_text(raw_text: str) -> str:
    """
    ASR 识别结果后处理流水线：

    1. 清理首尾无用标点与空格
    2. 智能家居热词纠错（同音错别字替换）
    3. 多余空格规范化

    Args:
        raw_text: ASR 模型输出的原始文本

    Returns:
        清洗后的文本
    """
    if not raw_text:
        return ""

    text = raw_text

    # Step 1: 清理首尾标点和空格（中文+英文）
    text = text.strip().strip(CJK_PUNCTUATION + ",.!?;: ")
    # 额外清理：去掉首尾的单个零宽字符或不可见字符
    text = text.strip("​‌‍﻿")

    if not text:
        return ""

    # Step 2: 热词纠错（同音错别字替换）
    # 先按长度降序排序，长词优先匹配
    sorted_corrections = sorted(HOTWORD_CORRECTIONS, key=lambda x: len(x[0]), reverse=True)
    for wrong, correct in sorted_corrections:
        # 使用替换而非简单的 str.replace，避免部分匹配导致误替换
        # 例如 "等" 替换为 "灯" 时，"等待" 不应变成 "灯待"
        text = _safe_replace(text, wrong, correct)

    # Step 3: 清理连续空格
    text = re.sub(r"\s+", "", text)  # ASR 中文结果通常不需要空格

    # Step 4: 再次清理可能因替换引入的首尾标点
    text = text.strip().strip(CJK_PUNCTUATION + ",.!?;: ")

    return text


def _safe_replace(text: str, old: str, new: str) -> str:
    """
    安全替换：仅在 old 作为独立分词出现时替换，避免部分匹配。

    对于中文上下文，如果 old 长度 >= 2，直接替换（中文无空格分词，
    过强的边界检测会导致 "开灯" 中的 "灯" 无法被替换）。
    对于长度为 1 的单字，加前后边界检测。
    """
    if len(old) >= 2:
        # 多字词：直接替换，中文同音词通常独立出现
        return text.replace(old, new)

    # 单字：使用负向零宽断言检测前后都不是中文字符/字母/数字才替换
    # 中文字符 Unicode 范围: 一-鿿
    han_pattern = re.compile(
        r"(?<![一-鿿\w])" + re.escape(old) + r"(?![一-鿿\w])"
    )
    return han_pattern.sub(new, text)


# ---------------------------------------------------------------------------
# /api/v1/asr 接口
# ---------------------------------------------------------------------------

@app.post("/api/v1/asr")
async def speech_to_text(file: UploadFile = File(...)):
    """
    语音识别接口：接收上传的音频文件，返回 ASR 识别文本。

    流程：
    1. 文件有效性校验（扩展名、大小、音频时长）
    2. ffmpeg 统一转码为 16kHz / 单声道 / 16bit PCM WAV
    3. SenseVoice 模型推理
    4. 文本后处理（热词纠错 + 标点清洗）

    Returns:
        {"code": 200, "msg": "success", "text": "识别结果"}
        {"code": 4xx/5xx, "msg": "错误描述", "text": ""}
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="未上传有效的音频文件")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {ext}（支持格式: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}）",
        )

    task_id = str(uuid.uuid4())
    safe_ext = ext or ".wav"
    raw_path = f"temp_raw_{task_id}{safe_ext}"
    converted_path = f"temp_16k_{task_id}.wav"

    try:
        content = await file.read()

        # ---- 文件有效性校验 ----
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="音频文件为空")

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"音频文件过大（{len(content) / 1024 / 1024:.1f}MB > 100MB）",
            )

        if len(content) < MIN_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="音频文件过小，疑似无效录音",
            )

        # 将上传内容写入临时文件
        with open(raw_path, "wb") as f:
            f.write(content)

        # ---- 音频有效性校验（时长检测） ----
        duration = 0.0
        try:
            # 尝试在转码前获取原始音频时长
            duration = _probe_duration(raw_path)
        except Exception:
            pass

        if 0 < duration < MIN_AUDIO_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"音频时长过短（{duration:.2f}秒 < {MIN_AUDIO_DURATION_SEC}秒），请录制更长的语音",
            )

        # ---- ffmpeg 统一转码 ----
        try:
            duration = convert_to_16k_wav(raw_path, converted_path)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"音频格式转换失败: {str(e)}",
            )

        # ---- 转码后二次时长校验 ----
        if 0 < duration < MIN_AUDIO_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"音频时长过短（{duration:.2f}秒），无法识别有效语音",
            )

        # ---- SenseVoice 模型推理 ----
        res = model.generate(
            input=converted_path,
            cache={},
            language="zh",       # 指定中文
            use_itn=True,        # 启用逆文本正则化（数字、标点标准化）
        )

        if res and len(res) > 0:
            raw_text = res[0].get("text", "").strip()

            if raw_text:
                # 文本后处理：热词纠错 + 标点清洗
                cleaned_text = post_process_text(raw_text)

                if cleaned_text:
                    print(
                        f"[ASR] 原始: '{raw_text[:60]}' → 清洗: '{cleaned_text[:60]}'"
                    )
                    return {
                        "code": 200,
                        "msg": "success",
                        "text": cleaned_text,
                    }

            return {"code": 500, "msg": "识别失败，未提取到有效音频信号", "text": ""}

        return {"code": 500, "msg": "识别失败，模型返回空结果", "text": ""}

    except HTTPException:
        raise
    except Exception as e:
        return {
            "code": 500,
            "msg": f"服务内部错误: {str(e)}",
            "text": "",
        }
    finally:
        # 确保临时文件被彻底删除，无论成功与否
        for p in [raw_path, converted_path]:
            if os.path.exists(p):
                for _ in range(3):  # 最多重试 3 次删除
                    try:
                        os.remove(p)
                        break
                    except PermissionError:
                        # Windows 下文件可能被短暂占用，等待后重试
                        import time as _time
                        _time.sleep(0.05)
                    except Exception:
                        break


# ---------------------------------------------------------------------------
# 独立启动入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)