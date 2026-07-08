from __future__ import annotations

import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from utils_core import (
    WORK_ROOT,
    byte_accuracy,
    byte_distance,
    bytes_to_preview_text,
    detect_domain,
    fmt_bytes,
    global_ssim_array,
    psnr_from_mse,
    sha256_bytes,
    string_accuracy,
    string_distance,
    write_temp_file,
)


def _row(group: str, metric: str, original: Any, decoded: Any, value: Any, note: str = "") -> Dict[str, Any]:
    return {"Group": group, "Metric": metric, "Original": original, "Decoded / recovered": decoded, "Value": value, "Note": note}


def build_raw_representation(path: str, data: bytes, *, domain: str, options: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any], str]:
    """Return raw bytes, raw metadata, and preview/output path for the chosen representation."""
    domain = str(domain or detect_domain(path, data))
    if domain == "image":
        return image_to_raw(data, options)
    if domain == "text":
        return text_to_raw(data, options)
    if domain == "audio":
        return audio_to_raw(path, data, options)
    raise ValueError("This first Raw Representation app supports image, text, and audio only.")


def image_to_raw(data: bytes, options: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any], str]:
    if Image is None:
        raise RuntimeError("Pillow is required for image raw representation.")
    mode_label = str(options.get("image_mode", "RGB pixels"))
    threshold = int(options.get("threshold", 128))
    max_side = int(options.get("max_side", 0) or 0)
    img = Image.open(io.BytesIO(bytes(data or b"")))
    img.load()
    if max_side > 0:
        img.thumbnail((max_side, max_side))
    if mode_label == "RGB pixels":
        raw_img = img.convert("RGB")
        raw_mode = "RGB"
        channels = 3
    elif mode_label == "Grayscale pixels":
        raw_img = img.convert("L")
        raw_mode = "L"
        channels = 1
    elif mode_label == "Binary pixels":
        gray = img.convert("L")
        raw_img = gray.point(lambda p: 255 if p >= threshold else 0).convert("L")
        raw_mode = "L"
        channels = 1
    else:
        raise ValueError(f"Unknown image raw mode: {mode_label}")
    raw = raw_img.tobytes()
    meta = {
        "domain": "image",
        "representation": mode_label,
        "raw_mode": raw_mode,
        "width": int(raw_img.width),
        "height": int(raw_img.height),
        "channels": int(channels),
        "dtype": "uint8",
        "threshold": threshold if mode_label == "Binary pixels" else None,
        "expected_raw_bytes": len(raw),
        "output_ext": ".png",
        "sha256_raw": sha256_bytes(raw),
    }
    preview = rebuild_raw_to_file(raw, meta, preferred_name="raw_image_preview")
    return raw, meta, preview


def text_to_raw(data: bytes, options: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any], str]:
    normalize = bool(options.get("normalize_line_endings", True))
    text = bytes(data or b"").decode("utf-8", errors="replace")
    if normalize:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = text.encode("utf-8")
    words = [w for w in text.replace("\n", " ").split(" ") if w]
    meta = {
        "domain": "text",
        "representation": "UTF-8 text bytes" + (" + normalized line endings" if normalize else ""),
        "encoding": "utf-8",
        "normalize_line_endings": normalize,
        "characters": len(text),
        "words": len(words),
        "lines": text.count("\n") + (1 if text else 0),
        "expected_raw_bytes": len(raw),
        "output_ext": ".txt",
        "sha256_raw": sha256_bytes(raw),
    }
    preview = rebuild_raw_to_file(raw, meta, preferred_name="raw_text_preview")
    return raw, meta, preview


def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def _convert_audio_with_ffmpeg(path: str, *, sample_rate: int | None = None, channels: int | None = None) -> bytes | None:
    exe = _ffmpeg_bin()
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "audio_pcm16.wav"
        cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-i", str(path)]
        if sample_rate and sample_rate > 0:
            cmd += ["-ar", str(int(sample_rate))]
        if channels and channels > 0:
            cmd += ["-ac", str(int(channels))]
        cmd += ["-acodec", "pcm_s16le", str(out)]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        if p.returncode != 0 or not out.exists():
            return None
        return out.read_bytes()


def _read_wav_pcm16(wav_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    with wave.open(io.BytesIO(bytes(wav_bytes or b"")), "rb") as w:
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_rate = w.getframerate()
        frames = w.getnframes()
        pcm = w.readframes(frames)
    if sample_width != 2:
        raise ValueError("Audio must be PCM16. Use FFmpeg conversion or upload a PCM16 WAV file.")
    meta = {
        "domain": "audio",
        "representation": "PCM16 waveform bytes",
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "sample_width": 2,
        "dtype": "int16",
        "frames": int(frames),
        "samples_total": int(frames) * int(channels),
        "duration": float(frames / float(sample_rate or 1)),
        "expected_raw_bytes": len(pcm),
        "output_ext": ".wav",
        "sha256_raw": sha256_bytes(pcm),
    }
    return pcm, meta


def audio_to_raw(path: str, data: bytes, options: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any], str]:
    sample_rate_opt = str(options.get("sample_rate", "Original"))
    channels_opt = str(options.get("channels", "Original"))
    sample_rate = None if sample_rate_opt == "Original" else int(sample_rate_opt)
    channels = None if channels_opt == "Original" else (1 if channels_opt == "Mono" else 2)

    wav_bytes = None
    # Prefer FFmpeg because it normalizes MP3/FLAC/OGG/WAV to PCM16.
    if path and os.path.exists(path):
        wav_bytes = _convert_audio_with_ffmpeg(path, sample_rate=sample_rate, channels=channels)
    if wav_bytes is None:
        # Fallback: only PCM16 WAV works here.
        wav_bytes = bytes(data or b"")
    pcm, meta = _read_wav_pcm16(wav_bytes)
    meta["sample_rate_option"] = sample_rate_opt
    meta["channels_option"] = channels_opt
    meta["ffmpeg_used"] = bool(_ffmpeg_bin() and path and os.path.exists(path))
    preview = rebuild_raw_to_file(pcm, meta, preferred_name="raw_audio_preview")
    return pcm, meta, preview


def _fit_length(data: bytes, expected: int) -> Tuple[bytes, str]:
    data = bytes(data or b"")
    expected = int(expected or len(data))
    if len(data) < expected:
        return data + bytes(expected - len(data)), f"Decoded raw bytes were shorter; padded {expected - len(data)} zero bytes."
    if len(data) > expected:
        return data[:expected], f"Decoded raw bytes were longer; truncated {len(data) - expected} bytes."
    return data, "Exact raw length."


def rebuild_raw_to_file(raw_bytes: bytes, meta: Dict[str, Any], *, preferred_name: str = "rebuilt") -> str:
    domain = str(meta.get("domain", ""))
    raw, _note = _fit_length(raw_bytes, int(meta.get("expected_raw_bytes", len(raw_bytes or b""))))
    if domain == "image":
        if Image is None:
            raise RuntimeError("Pillow is required to rebuild image raw data.")
        mode = str(meta.get("raw_mode", "RGB"))
        size = (int(meta.get("width", 1)), int(meta.get("height", 1)))
        img = Image.frombytes(mode, size, raw)
        out = io.BytesIO()
        img.save(out, format="PNG")
        return write_temp_file(out.getvalue(), preferred_name=preferred_name, ext=".png")
    if domain == "text":
        return write_temp_file(raw, preferred_name=preferred_name, ext=".txt")
    if domain == "audio":
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(int(meta.get("channels", 1)))
            w.setsampwidth(int(meta.get("sample_width", 2)))
            w.setframerate(int(meta.get("sample_rate", 16000)))
            w.writeframes(raw)
        return write_temp_file(out.getvalue(), preferred_name=preferred_name, ext=".wav")
    return write_temp_file(raw, preferred_name=preferred_name, ext=".bin")


def raw_quality_rows(original_raw: bytes, decoded_raw: bytes, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    domain = str(meta.get("domain", ""))
    expected = int(meta.get("expected_raw_bytes", len(original_raw or b"")))
    decoded_fit, length_note = _fit_length(decoded_raw, expected)
    rows: List[Dict[str, Any]] = [
        _row("Raw recovery", "Representation", meta.get("representation", "—"), meta.get("representation", "—"), "Same"),
        _row("Raw recovery", "Raw size", fmt_bytes(len(original_raw or b"")), fmt_bytes(len(decoded_raw or b"")), "Match" if len(original_raw or b"") == len(decoded_raw or b"") else "Different", length_note),
        _row("Raw recovery", "Raw byte accuracy", "1.000", "—", f"{byte_accuracy(original_raw, decoded_fit):.6f}"),
        _row("Raw recovery", "Raw byte mismatches", 0, "—", byte_distance(original_raw, decoded_fit)),
        _row("Raw recovery", "SHA256 raw match", "Yes", "—", "Yes" if sha256_bytes(original_raw) == sha256_bytes(decoded_fit) else "No"),
    ]
    if domain == "image":
        rows.extend(image_quality_rows(original_raw, decoded_fit, meta))
    elif domain == "text":
        rows.extend(text_quality_rows(original_raw, decoded_fit))
    elif domain == "audio":
        rows.extend(audio_quality_rows(original_raw, decoded_fit, meta))
    return rows


def image_quality_rows(original: bytes, decoded: bytes, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    if np is None:
        return [_row("Image quality", "Image metrics", "—", "—", "Not available", "Install numpy.")]
    w, h = int(meta.get("width", 1)), int(meta.get("height", 1))
    channels = int(meta.get("channels", 1))
    a = np.frombuffer(bytes(original or b""), dtype=np.uint8)
    b = np.frombuffer(bytes(decoded or b""), dtype=np.uint8)
    n = min(len(a), len(b), w * h * channels)
    a = a[:n]
    b = b[:n]
    if n == 0:
        return [_row("Image quality", "Image metrics", "—", "—", "Not available", "Empty image raw data.")]
    diff = a.astype("float32") - b.astype("float32")
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    psnr = psnr_from_mse(mse, peak=255.0)
    channel_acc = float(np.mean(a == b))
    pixel_acc = channel_acc
    ssim_note = "Global SSIM approximation."
    try:
        if channels == 1:
            arr_a = a.reshape((h, w))
            arr_b = b.reshape((h, w))
        else:
            arr_a = a.reshape((h, w, channels))
            arr_b = b.reshape((h, w, channels))
        ssim = global_ssim_array(arr_a, arr_b, data_range=255.0)
        if channels > 1:
            pixel_acc = float(np.mean(np.all(arr_a == arr_b, axis=2)))
    except Exception as exc:
        ssim = float("nan")
        ssim_note = str(exc)
    return [
        _row("Image quality", "Image size", f"{w}×{h}×{channels}", f"{w}×{h}×{channels}", "Same"),
        _row("Image quality", "MSE", "0", "—", f"{mse:.6f}", "Lower is better."),
        _row("Image quality", "MAE", "0", "—", f"{mae:.6f}", "Lower is better."),
        _row("Image quality", "PSNR", "∞", "—", f"{psnr:.3f} dB", "Higher is better."),
        _row("Image quality", "SSIM", "1.000", "—", f"{ssim:.6f}", ssim_note),
        _row("Image quality", "Exact pixel accuracy", "1.000", "—", f"{pixel_acc:.6f}"),
        _row("Image quality", "Channel accuracy", "1.000", "—", f"{channel_acc:.6f}"),
    ]


def text_quality_rows(original: bytes, decoded: bytes) -> List[Dict[str, Any]]:
    o = bytes_to_preview_text(original, limit=max(len(original or b""), 1_000_000))
    d = bytes_to_preview_text(decoded, limit=max(len(decoded or b""), 1_000_000))
    ow = [w for w in o.replace("\n", " ").split(" ") if w]
    dw = [w for w in d.replace("\n", " ").split(" ") if w]
    n = min(len(ow), len(dw))
    word_diff = sum(1 for i in range(n) if ow[i] != dw[i]) + abs(len(ow) - len(dw))
    word_acc = 1.0 - word_diff / max(len(ow), len(dw), 1)
    return [
        _row("Text quality", "Characters", len(o), len(d), "Match" if len(o) == len(d) else "Different"),
        _row("Text quality", "Character accuracy", "1.000", "—", f"{string_accuracy(o, d):.6f}"),
        _row("Text quality", "Character differences", 0, "—", string_distance(o, d)),
        _row("Text quality", "Words", len(ow), len(dw), "Match" if len(ow) == len(dw) else "Different"),
        _row("Text quality", "Word accuracy", "1.000", "—", f"{word_acc:.6f}"),
        _row("Text quality", "Exact text match", "Yes", "—", "Yes" if o == d else "No"),
    ]


def audio_quality_rows(original: bytes, decoded: bytes, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    if np is None:
        return [_row("Audio quality", "Waveform metrics", "—", "—", "Not available", "Install numpy.")]
    a = np.frombuffer(bytes(original or b""), dtype=np.int16).astype("float64")
    b = np.frombuffer(bytes(decoded or b""), dtype=np.int16).astype("float64")
    n = min(len(a), len(b))
    if n == 0:
        return [_row("Audio quality", "Waveform metrics", "—", "—", "Not available", "Empty waveform.")]
    a = a[:n]
    b = b[:n]
    err = a - b
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    psnr = psnr_from_mse(mse, peak=32767.0)
    signal_power = float(np.mean(a ** 2))
    snr = 99.0 if mse <= 1e-12 else float(10.0 * math.log10(max(signal_power, 1e-12) / mse))
    sample_acc = float(np.mean(a == b))
    return [
        _row("Audio quality", "Duration", f"{float(meta.get('duration', 0)):.3f} s", f"{float(meta.get('duration', 0)):.3f} s", "Same after length fit"),
        _row("Audio quality", "Sample rate", meta.get("sample_rate"), meta.get("sample_rate"), "Same"),
        _row("Audio quality", "Channels", meta.get("channels"), meta.get("channels"), "Same"),
        _row("Audio quality", "Waveform RMSE", "0", "—", f"{rmse:.6f}"),
        _row("Audio quality", "Waveform MAE", "0", "—", f"{mae:.6f}"),
        _row("Audio quality", "Waveform PSNR", "∞", "—", f"{psnr:.3f} dB"),
        _row("Audio quality", "Waveform SNR", "∞", "—", f"{snr:.3f} dB"),
        _row("Audio quality", "Exact sample accuracy", "1.000", "—", f"{sample_acc:.6f}"),
    ]


def raw_meta_summary(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    domain = str(meta.get("domain", ""))
    common = [
        {"Property": "Raw domain", "Value": domain},
        {"Property": "Raw representation", "Value": meta.get("representation", "—")},
        {"Property": "Raw size", "Value": fmt_bytes(meta.get("expected_raw_bytes"))},
        {"Property": "SHA256 raw", "Value": str(meta.get("sha256_raw", ""))[:16] + "..."},
    ]
    if domain == "image":
        common.extend([
            {"Property": "Image size", "Value": f"{meta.get('width')}×{meta.get('height')}"},
            {"Property": "Channels", "Value": meta.get("channels")},
            {"Property": "Mode", "Value": meta.get("raw_mode")},
        ])
    elif domain == "text":
        common.extend([
            {"Property": "Characters", "Value": meta.get("characters")},
            {"Property": "Words", "Value": meta.get("words")},
            {"Property": "Lines", "Value": meta.get("lines")},
        ])
    elif domain == "audio":
        common.extend([
            {"Property": "Duration", "Value": f"{float(meta.get('duration', 0)):.3f} s"},
            {"Property": "Sample rate", "Value": meta.get("sample_rate")},
            {"Property": "Channels", "Value": meta.get("channels")},
            {"Property": "Frames", "Value": meta.get("frames")},
        ])
    return common
