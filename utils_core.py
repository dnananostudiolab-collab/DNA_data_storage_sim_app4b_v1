from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

WORK_ROOT = Path(tempfile.gettempdir()) / "raw_representation_dna_app"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

TEXT_EXTENSIONS = {".txt", ".text", ".md", ".csv", ".tsv", ".json", ".xml", ".html", ".htm", ".py", ".log", ".yaml", ".yml"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}


def safe_basename(name: str) -> str:
    name = os.path.basename(str(name or "file.bin"))
    out = []
    for ch in name:
        if ch.isalnum() or ch in "._- ()":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip() or "file.bin"


def fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "—"
    try:
        x = float(n)
    except Exception:
        return "—"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024.0 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TB"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(bytes(data or b"")).hexdigest()


def bytes_to_bitstring(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in bytes(data or b""))


def bitstring_to_bytes(bits: str, *, pad_to_byte: bool = True) -> tuple[bytes, int]:
    bits = "".join(ch for ch in str(bits or "") if ch in "01")
    pad = 0
    if pad_to_byte and len(bits) % 8:
        pad = 8 - (len(bits) % 8)
        bits += "0" * pad
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        if len(chunk) == 8:
            out.append(int(chunk, 2))
    return bytes(out), pad


def bytes_to_preview_text(data: bytes, limit: int = 12000) -> str:
    return bytes(data or b"").decode("utf-8", errors="replace")[:int(limit)]


def _can_decode_as_text(data: bytes) -> bool:
    data = bytes(data or b"")
    if not data:
        return True
    try:
        text = data[:8192].decode("utf-8")
    except Exception:
        return False
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\r\n\t")
    return control <= max(1, len(text) // 100)


def detect_domain(name: str, data: bytes) -> str:
    ext = Path(str(name or "")).suffix.lower()
    head = bytes(data or b"")[:64]
    if ext in IMAGE_EXTENSIONS or head.startswith(b"\x89PNG\r\n\x1a\n") or head.startswith(b"\xff\xd8\xff") or head.startswith(b"BM") or head.startswith((b"GIF87a", b"GIF89a")):
        return "image"
    if ext in AUDIO_EXTENSIONS or (head.startswith(b"RIFF") and head[8:12] == b"WAVE") or head.startswith(b"ID3") or head.startswith(b"fLaC") or head.startswith(b"OggS"):
        return "audio"
    if ext in TEXT_EXTENSIONS or _can_decode_as_text(data):
        return "text"
    return "unsupported"


def byte_distance(a: bytes, b: bytes) -> int:
    a = bytes(a or b"")
    b = bytes(b or b"")
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def byte_accuracy(a: bytes, b: bytes) -> float:
    denom = max(len(bytes(a or b"")), len(bytes(b or b"")), 1)
    return 1.0 - byte_distance(a, b) / denom


def string_distance(a: str, b: str) -> int:
    a = str(a or "")
    b = str(b or "")
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def string_accuracy(a: str, b: str) -> float:
    denom = max(len(str(a or "")), len(str(b or "")), 1)
    return 1.0 - string_distance(a, b) / denom


def hamming_distance_str(a: str, b: str) -> int:
    return string_distance(a, b)


def write_temp_file(data: bytes, preferred_name: str = "output", ext: str = ".bin") -> str:
    out_dir = WORK_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ext.startswith("."):
        ext = "." + ext
    sig = hashlib.sha1(bytes(data or b"")[:4096]).hexdigest()[:10]
    path = out_dir / f"{preferred_name}_{sig}{ext}"
    path.write_bytes(bytes(data or b""))
    return str(path)


_PREVIEW_WIDGET_COUNTER = 0


def preview_file_streamlit(st, path: str, title: str = "Preview", *, key_suffix: str = "") -> None:
    """Preview a file with a collision-safe Streamlit widget key.

    Text previews use `st.text_area`, which requires a globally unique key.
    In this app the same text file can appear in Panel 5 and Panel 6, and the
    same helper can also be called inside both No-ECC and ECC tabs.  A stable
    hash alone can collide when the same path/title/context is rendered twice,
    so we include a per-rerun counter in the key.
    """
    global _PREVIEW_WIDGET_COUNTER
    _PREVIEW_WIDGET_COUNTER += 1

    st.markdown(f"#### {title}")
    if not path or not os.path.exists(path):
        st.info("Preview is not available.")
        return
    data = Path(path).read_bytes()
    ext = Path(path).suffix.lower()
    key_base = f"{path}|{title}|{key_suffix}|{len(data)}|{_PREVIEW_WIDGET_COUNTER}"
    preview_key = "preview_" + hashlib.sha1(key_base.encode("utf-8", errors="ignore")).hexdigest()[:16]
    try:
        if ext in IMAGE_EXTENSIONS:
            st.image(path, width=260)
        elif ext == ".wav" or ext in AUDIO_EXTENSIONS:
            st.audio(path)
        elif ext in TEXT_EXTENSIONS or _can_decode_as_text(data):
            st.text_area("Text preview", bytes_to_preview_text(data, 20000), height=260, label_visibility="collapsed", key=preview_key)
        else:
            st.info("Preview is not available for this file type.")
    except Exception as exc:
        st.warning(f"Preview failed: {exc}")


def psnr_from_mse(mse: float, peak: float = 255.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(float(peak) / math.sqrt(float(mse))))


def global_ssim_array(a, b, data_range: float = 255.0) -> float:
    if np is None:
        return float("nan")
    x = np.asarray(a).astype("float64")
    y = np.asarray(b).astype("float64")
    if x.shape != y.shape:
        raise ValueError("arrays must have same shape")
    if x.ndim == 2:
        x = x[:, :, None]
        y = y[:, :, None]
    vals = []
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    for ch in range(x.shape[2]):
        xx = x[:, :, ch].reshape(-1)
        yy = y[:, :, ch].reshape(-1)
        mux, muy = float(xx.mean()), float(yy.mean())
        vx, vy = float(xx.var()), float(yy.var())
        cov = float(((xx - mux) * (yy - muy)).mean())
        vals.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))
    return float(np.mean(vals)) if vals else float("nan")
