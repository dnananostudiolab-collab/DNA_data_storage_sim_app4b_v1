from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from dna_design import (
    MAPPING_OPTIONS,
    clean_dna,
    decode_dna_to_bytes,
    display_mapping,
    encode_bytes_to_dna,
    gc_content,
    homopolymer_stats,
)
from error_model import mutate_dna
from raw_codec import (
    build_raw_representation,
    raw_meta_summary,
    raw_quality_rows,
    rebuild_raw_to_file,
)
from rs_binary_ecc import encode_rs_bytes, decode_rs_bytes
from utils_core import (
    WORK_ROOT,
    byte_accuracy,
    byte_distance,
    bytes_to_bitstring,
    bytes_to_preview_text,
    detect_domain,
    fmt_bytes,
    preview_file_streamlit,
    safe_basename,
    sha256_bytes,
)

APP_STEPS = [
    (1, "Input"),
    (2, "Raw representation"),
    (3, "Encoding"),
    (4, "Strand Design"),
    (5, "Decoding"),
    (6, "Summarization"),
]

PANEL_TITLES = {
    "input": "Input",
    "raw": "Raw representation",
    "dna": "Encoding",
    "strand": "Strand Design",
    "decode": "Decoding",
    "summary": "Summarization",
}

DEFAULT_FBR = "ACACGACGCTCTTCCGATCT"
DEFAULT_RBR = "AGATCGGAAGAGCACACGTCT"
REGION_COLORS = {
    "FBR": ("#DBEAFE", "#1E3A8A"),
    "Index": ("#EDE9FE", "#4C1D95"),
    "Payload": ("#DCFCE7", "#14532D"),
    "Filler": ("#F1F5F9", "#475569"),
    "RBR": ("#FFEDD5", "#7C2D12"),
}


def apply_app_style() -> None:
    st.markdown(
        """
<style>
:root {
  --bg:#F8FAFC; --surface:#FFFFFF; --border:#D8E1EC; --text:#0F172A;
  --muted:#64748B; --primary:#2563EB; --primary-soft:#DBEAFE;
}
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 1.2rem; max-width: 1300px; }
.hero-card {
  background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%);
  border: 1px solid var(--border); border-radius: 18px; padding: 1.1rem 1.25rem;
  margin-bottom: 1rem; box-shadow: 0 10px 30px rgba(15,23,42,0.045);
}
.hero-title { font-size: 24px; font-weight: 780; letter-spacing: -0.02em; }
.hero-subtitle { color: var(--muted); font-size: 15px; margin-top: 0.25rem; }
.step-heading { display:flex; align-items:center; gap:0.65rem; margin: 0.1rem 0 0.8rem 0; }
.step-badge {
  width: 30px; height: 30px; border-radius: 999px; background: var(--primary); color:white;
  display:inline-flex; align-items:center; justify-content:center; font-weight:760;
}
.step-title { font-size: 20px; font-weight: 760; }
.pipeline-steps { display:grid; grid-template-columns: repeat(6, 1fr); gap:0.5rem; margin-bottom:1rem; }
.pipeline-step { border:1px solid var(--border); background:#fff; border-radius:14px; padding:0.65rem; }
.pipeline-step.done { background:#DCFCE7; border-color:#86EFAC; }
.pipeline-step.current { background:#DBEAFE; border-color:#93C5FD; }
.step-num { font-weight:760; margin-right:0.35rem; color:#1E3A8A; }
.step-name { font-weight:650; font-size:13px; }
.step-state { font-size:12px; color:var(--muted); margin-top:0.15rem; }
.region-tag {
  display:inline-block; padding:0.35rem 0.55rem; border-radius:12px; margin:0.1rem 0.2rem 0.1rem 0;
  font-family: Consolas, monospace; font-size:12px; line-height:1.6; word-break:break-all;
}
.error-base { background:#FECACA; color:#7F1D1D; font-weight:800; padding:0 1px; border-radius:3px; }
.small-note { color:#64748B; font-size:13px; }
</style>
""",
        unsafe_allow_html=True,
    )


ACTIVE_PREFIX = "noecc"
ACTIVE_ECC_ENABLED = False


def _set_pipeline_context(prefix: str, ecc_enabled: bool) -> None:
    global ACTIVE_PREFIX, ACTIVE_ECC_ENABLED
    ACTIVE_PREFIX = str(prefix or "pipe")
    ACTIVE_ECC_ENABLED = bool(ecc_enabled)


def _key(name: str) -> str:
    return f"raw_app_{ACTIVE_PREFIX}_{name}"

def _content_key(name: str, value: Any) -> str:
    """Return a widget key that changes when preview content changes.

    Streamlit keeps text_area contents by widget key. If the key is fixed,
    changing SM ↔ R∞ can update metrics while the Base string text_area still
    shows the previous DNA preview. Including a content hash forces refresh.
    """
    if isinstance(value, bytes):
        raw = value
    else:
        raw = str(value or "").encode("utf-8", errors="ignore")

    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"{_key(name)}_{digest}"

def step_header(number: int, title: str) -> None:
    st.markdown(
        f"""
<div class="step-heading">
  <span class="step-badge">{number}</span>
  <span class="step-title">{title}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def _metrics_row(items: List[tuple[str, Any]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def _download_text_button(label: str, text: str, file_name: str, *, key: str) -> None:
    st.download_button(label, data=str(text or "").encode("utf-8"), file_name=file_name, mime="text/plain", use_container_width=True, key=key)


def _download_bytes_button(label: str, data: bytes, file_name: str, *, key: str) -> None:
    st.download_button(label, data=bytes(data or b""), file_name=file_name, mime="application/octet-stream", use_container_width=True, key=key)


def _clear_downstream(start: str = "raw") -> None:
    groups = {
        "raw": ["raw_bytes", "raw_meta", "raw_preview_path"],
        "dna": ["dna", "bits", "codec_meta", "encoding_bytes", "ecc_meta", "encode_input_sha", "encode_signature"],
        "strand": ["strand_rows", "strand_signature", "noisy_strand_rows", "noisy_dna", "error_events", "error_metrics"],
        "decode": ["decoded_payload_bytes", "decoded_raw_bytes", "decoded_bits", "decoded_meta", "ecc_repair_report", "rebuilt_path", "decode_note"],
    }
    order = ["raw", "dna", "strand", "decode"]
    for group in order[order.index(start):]:
        for name in groups[group]:
            st.session_state.pop(_key(name), None)


def _store_upload(uploaded) -> None:
    data = uploaded.getvalue()
    name = safe_basename(uploaded.name or "upload.bin")
    sig = f"{name}|{len(data)}|{sha256_bytes(data)}"
    if st.session_state.get(_key("input_signature")) == sig:
        return
    upload_dir = WORK_ROOT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / name
    path.write_bytes(data)
    st.session_state.update({
        _key("input_signature"): sig,
        _key("input_name"): name,
        _key("input_path"): str(path),
        _key("input_bytes"): data,
    })
    _clear_downstream("raw")


def _input_available() -> bool:
    return bool(st.session_state.get(_key("input_bytes")))


def _preview_seq(seq: str, n: int = 900) -> str:
    seq = clean_dna(seq)
    return seq[:n] + ("..." if len(seq) > n else "")


def _step_checks() -> Dict[int, bool]:
    return {
        1: _input_available(),
        2: bool(st.session_state.get(_key("raw_bytes"))),
        3: bool(st.session_state.get(_key("dna"))),
        4: bool(st.session_state.get(_key("noisy_dna"))),
        5: bool(st.session_state.get(_key("rebuilt_path"))),
        6: bool(st.session_state.get(_key("rebuilt_path"))),
    }


def render_stepper() -> None:
    checks = _step_checks()
    parts = ['<div class="pipeline-steps">']
    for n, label in APP_STEPS:
        css = "done" if checks.get(n) else ("current" if all(checks.get(i) for i in range(1, n)) else "")
        state = "Done" if checks.get(n) else ("Next" if css == "current" else "Waiting")
        parts.append(
            f'<div class="pipeline-step {css}"><div><span class="step-num">{n}</span>'
            f'<span class="step-name">{label}</span></div><div class="step-state">{state}</div></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Strand helpers
# -----------------------------------------------------------------------------


def _base4_index(n: int, length: int) -> str:
    n = max(0, int(n))
    chars = []
    for _ in range(max(0, int(length))):
        chars.append("ACGT"[n & 0b11])
        n >>= 2
    return "".join(reversed(chars))


def _make_filler(seed: int, length: int) -> str:
    if length <= 0:
        return ""
    bases = "ACGT"
    return "".join(bases[(seed + i) % 4] for i in range(length))


def _make_prepared_strands(dna: str, *, total_len: int, index_len: int, fbr: str, rbr: str) -> List[Dict[str, Any]]:
    dna = clean_dna(dna)
    fbr = clean_dna(fbr)
    rbr = clean_dna(rbr)
    fixed = len(fbr) + int(index_len) + len(rbr)
    payload_capacity = int(total_len) - fixed
    if payload_capacity <= 0:
        raise ValueError("Total strand length is too short for FBR + Index + RBR.")
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(dna), payload_capacity):
        no = len(rows) + 1
        payload = dna[start:start + payload_capacity]
        filler = _make_filler(no, payload_capacity - len(payload))
        index = _base4_index(no, int(index_len))
        full = fbr + index + payload + filler + rbr
        hp = homopolymer_stats(full)
        rows.append({
            "No.": no,
            "Type": "Prepared strand",
            "FBR": fbr,
            "Index": index,
            "Payload": payload,
            "Filler": filler,
            "RBR": rbr,
            "Full strand": full,
            "Index length": len(index),
            "Payload length": len(payload),
            "Filler length": len(filler),
            "Total length": len(full),
            "Payload capacity": payload_capacity,
            "Payload global start": start + 1,
            "Payload start in full": len(fbr) + len(index) + 1,
            "GC content": f"{gc_content(full):.3f}",
            "Longest homopolymer": hp.get("longest", 0),
        })
    return rows


def _row_regions(row: Dict[str, Any]) -> List[tuple[str, str]]:
    return [
        ("FBR", clean_dna(row.get("FBR", ""))),
        ("Index", clean_dna(row.get("Index", ""))),
        ("Payload", clean_dna(row.get("Payload", ""))),
        ("Filler", clean_dna(row.get("Filler", ""))),
        ("RBR", clean_dna(row.get("RBR", ""))),
    ]


def _region_html(name: str, seq: str, error_positions: set[int] | None = None, start_pos: int = 1) -> str:
    bg, fg = REGION_COLORS.get(name, ("#F8FAFC", "#0F172A"))
    error_positions = error_positions or set()
    chars = []
    for i, ch in enumerate(clean_dna(seq), start=start_pos):
        chars.append(f'<span class="error-base">{ch}</span>' if i in error_positions else ch)
    body = "".join(chars) if chars else "—"
    return f'<span class="region-tag" style="background:{bg};color:{fg};"><b>{name}</b>: {body}</span>'


def _render_segmented_strand(row: Dict[str, Any], title: str, error_positions: set[int] | None = None) -> None:
    parts = []
    cursor = 1
    for name, seq in _row_regions(row):
        parts.append(_region_html(name, seq, error_positions, cursor))
        cursor += len(clean_dna(seq))
    st.markdown(f"**{title}**", unsafe_allow_html=True)
    st.markdown("".join(parts), unsafe_allow_html=True)


def _strand_summary(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    keep = ["No.", "Type", "Index length", "Payload length", "Filler length", "Total length", "GC content", "Longest homopolymer"]
    return pd.DataFrame([{k: r.get(k, "—") for k in keep} for r in rows])


def _mutate_prepared_rows(rows: List[Dict[str, Any]], *, scope: str, substitution_rate: float, insertion_rate: float, deletion_rate: float, seed: int, allow_indels: bool) -> tuple[List[Dict[str, Any]], str, List[Dict[str, Any]], Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []
    total = {"substitutions": 0, "insertions": 0, "deletions": 0, "total_errors": 0}
    noisy_payloads: List[str] = []
    for row in rows:
        no = int(row.get("No.", 0))
        row_seed = int(seed) + no * 1000003
        fbr = clean_dna(row.get("FBR", ""))
        idx = clean_dna(row.get("Index", ""))
        payload = clean_dna(row.get("Payload", ""))
        filler = clean_dna(row.get("Filler", ""))
        rbr = clean_dna(row.get("RBR", ""))
        payload_full_start = len(fbr) + len(idx) + 1
        payload_global_start = int(row.get("Payload global start", 1))
        if scope == "Payload only":
            noisy_payload, evs, m = mutate_dna(payload, substitution_rate=substitution_rate, insertion_rate=insertion_rate, deletion_rate=deletion_rate, seed=row_seed, allow_indels=allow_indels)
            new = dict(row)
            new["Payload"] = noisy_payload
            new["Full strand"] = fbr + idx + noisy_payload + filler + rbr
            for ev in evs:
                local_pos = int(ev.get("Original position", ev.get("Read position", 1)))
                ev2 = dict(ev)
                ev2.update({"Strand": no, "Region": "Payload", "Full-strand position": payload_full_start + local_pos - 1, "DNA payload position": payload_global_start + local_pos - 1})
                all_events.append(ev2)
            noisy_payloads.append(noisy_payload)
        else:
            full = fbr + idx + payload + filler + rbr
            noisy_full, evs, m = mutate_dna(full, substitution_rate=substitution_rate, insertion_rate=insertion_rate, deletion_rate=deletion_rate, seed=row_seed, allow_indels=allow_indels)
            p0 = len(fbr) + len(idx)
            noisy_payload = noisy_full[p0:p0 + len(payload)]
            new = dict(row)
            new["Payload"] = noisy_payload
            new["Full strand"] = noisy_full
            for ev in evs:
                pos = int(ev.get("Original position", ev.get("Read position", 1)))
                region = "Payload" if p0 < pos <= p0 + len(payload) else "Non-payload"
                ev2 = dict(ev)
                ev2.update({"Strand": no, "Region": region, "Full-strand position": pos, "DNA payload position": payload_global_start + (pos - p0) - 1 if region == "Payload" else "—"})
                all_events.append(ev2)
            noisy_payloads.append(noisy_payload)
        for k in total:
            total[k] += int(m.get(k, 0))
        out_rows.append(new)
    return out_rows, "".join(noisy_payloads), all_events, total




# -----------------------------------------------------------------------------
# Summary helpers
# -----------------------------------------------------------------------------

def _render_property_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        st.info("No summary data available.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _short_sha(data: bytes) -> str:
    return sha256_bytes(data)[:12] + "..." if data is not None else "—"


def _basic_file_rows(path: str, data: bytes, *, label: str = "File") -> List[Dict[str, Any]]:
    return [
        {"Property": "Data", "Value": label},
        {"Property": "Size", "Value": fmt_bytes(len(data or b""))},
        {"Property": "SHA256", "Value": _short_sha(data or b"")},
        {"Property": "Path", "Value": Path(path).name if path else "—"},
    ]


def _raw_encoded_rows(raw_meta: Dict[str, Any], raw_bytes: bytes, encoding_bytes: bytes, dna: str, mapping: str, ecc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        {"Property": "Representation", "Value": raw_meta.get("representation", "—")},
        {"Property": "Raw domain", "Value": raw_meta.get("domain", "—")},
        {"Property": "Raw size", "Value": fmt_bytes(len(raw_bytes or b""))},
        {"Property": "DNA design rule", "Value": display_mapping(mapping)},
        {"Property": "ECC option", "Value": "Reed-Solomon" if ACTIVE_ECC_ENABLED else "None"},
        {"Property": "Encoded bytes", "Value": fmt_bytes(len(encoding_bytes or b""))},
        {"Property": "DNA length", "Value": f"{len(clean_dna(dna)):,} nt"},
    ]
    if ACTIVE_ECC_ENABLED:
        rows.extend([
            {"Property": "ECC overhead", "Value": f"{float(ecc_meta.get('ecc_overhead_ratio', 1.0)):.2f}×"},
            {"Property": "RS data/parity", "Value": f"{ecc_meta.get('data_block_size', '—')} + {ecc_meta.get('parity_bytes', '—')} bytes"},
        ])
    return rows


def _decoded_result_rows(raw_bytes: bytes, decoded_raw: bytes, raw_meta: Dict[str, Any], repair_report: Dict[str, Any], rebuilt_path: str) -> List[Dict[str, Any]]:
    sha_match = sha256_bytes(raw_bytes or b"") == sha256_bytes(decoded_raw or b"")
    rows = [
        {"Property": "Recovered raw size", "Value": fmt_bytes(len(decoded_raw or b""))},
        {"Property": "Expected raw size", "Value": fmt_bytes(raw_meta.get("expected_raw_bytes"))},
        {"Property": "Raw byte accuracy", "Value": f"{byte_accuracy(raw_bytes, decoded_raw):.6f}"},
        {"Property": "Raw byte mismatches", "Value": byte_distance(raw_bytes, decoded_raw)},
        {"Property": "SHA256 raw match", "Value": "Yes" if sha_match else "No"},
        {"Property": "Rebuilt output", "Value": "Yes" if rebuilt_path else "No"},
    ]
    if ACTIVE_ECC_ENABLED:
        rows.extend([
            {"Property": "RS repair success", "Value": "Yes" if repair_report.get("repair_success") else "No"},
            {"Property": "Corrected blocks", "Value": repair_report.get("corrected_blocks", 0)},
            {"Property": "Failed blocks", "Value": repair_report.get("failed_blocks", 0)},
            {"Property": "Corrected symbols", "Value": repair_report.get("corrected_symbols", 0)},
        ])
    else:
        rows.append({"Property": "RS repair success", "Value": "Not used"})
    return rows


def _encoding_statistics_rows(raw_meta: Dict[str, Any], raw_bytes: bytes, encoding_bytes: bytes, dna: str, strand_rows: List[Dict[str, Any]], mapping: str, ecc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    hp = homopolymer_stats(dna)
    rows = [
        {"Property": "Raw representation", "Value": raw_meta.get("representation", "—")},
        {"Property": "Raw bytes", "Value": fmt_bytes(len(raw_bytes or b""))},
        {"Property": "DNA design rule", "Value": display_mapping(mapping)},
        {"Property": "ECC option", "Value": "Reed-Solomon" if ACTIVE_ECC_ENABLED else "None"},
        {"Property": "Encoded bytes", "Value": fmt_bytes(len(encoding_bytes or b""))},
        {"Property": "DNA length", "Value": f"{len(clean_dna(dna)):,} nt"},
        {"Property": "Designed strands", "Value": len(strand_rows or [])},
        {"Property": "GC content", "Value": f"{gc_content(dna):.3f}"},
        {"Property": "Longest homopolymer", "Value": hp.get("longest", 0)},
    ]
    if ACTIVE_ECC_ENABLED:
        rows.append({"Property": "ECC overhead", "Value": f"{float(ecc_meta.get('ecc_overhead_ratio', 1.0)):.2f}×"})
    return rows


def _error_report_rows(dna: str, noisy: str, errors: Dict[str, Any]) -> List[Dict[str, Any]]:
    clean_len = len(clean_dna(dna))
    noisy_len = len(clean_dna(noisy))
    total_errors = int(errors.get("total_errors", 0) or 0)
    return [
        {"Property": "Clean DNA length", "Value": f"{clean_len:,} nt"},
        {"Property": "Noisy DNA length", "Value": f"{noisy_len:,} nt"},
        {"Property": "Added errors", "Value": total_errors},
        {"Property": "Substitutions", "Value": errors.get("substitutions", 0)},
        {"Property": "Insertions", "Value": errors.get("insertions", 0)},
        {"Property": "Deletions", "Value": errors.get("deletions", 0)},
        {"Property": "Observed DNA error rate", "Value": f"{total_errors / max(1, clean_len):.4%}"},
    ]


def _decode_recovery_rows(raw_bytes: bytes, encoding_bytes: bytes, decoded_payload: bytes, decoded_raw: bytes, repair_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        {"Property": "Decoded payload bytes", "Value": fmt_bytes(len(decoded_payload or b""))},
        {"Property": "Recovered raw bytes", "Value": fmt_bytes(len(decoded_raw or b""))},
        {"Property": "Final raw byte accuracy", "Value": f"{byte_accuracy(raw_bytes, decoded_raw):.6f}"},
        {"Property": "Final raw byte mismatches", "Value": byte_distance(raw_bytes, decoded_raw)},
        {"Property": "SHA256 raw match", "Value": "Yes" if sha256_bytes(raw_bytes or b"") == sha256_bytes(decoded_raw or b"") else "No"},
    ]
    if ACTIVE_ECC_ENABLED:
        rows.extend([
            {"Property": "Protected-byte accuracy before RS", "Value": f"{byte_accuracy(encoding_bytes, decoded_payload):.6f}"},
            {"Property": "RS repair success", "Value": "Yes" if repair_report.get("repair_success") else "No"},
            {"Property": "Corrected blocks", "Value": repair_report.get("corrected_blocks", 0)},
            {"Property": "Failed blocks", "Value": repair_report.get("failed_blocks", 0)},
        ])
    else:
        rows.append({"Property": "RS repair success", "Value": "Not used"})
    return rows


def _select_key_quality_rows(raw_bytes: bytes, decoded_raw: bytes, raw_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = raw_quality_rows(raw_bytes, decoded_raw, raw_meta)
    domain = str(raw_meta.get("domain", ""))
    keep_by_domain = {
        "image": {"Raw byte accuracy", "SHA256 raw match", "PSNR", "SSIM", "Exact pixel accuracy"},
        "text": {"Raw byte accuracy", "SHA256 raw match", "Character accuracy", "Word accuracy", "Exact text match"},
        "audio": {"Raw byte accuracy", "SHA256 raw match", "Waveform PSNR", "Waveform SNR", "Exact sample accuracy"},
    }
    keep = keep_by_domain.get(domain, {"Raw byte accuracy", "SHA256 raw match"})
    selected = [r for r in rows if str(r.get("Metric")) in keep]
    return selected or rows[:6]


def _final_summary_rows(raw_bytes: bytes, decoded_raw: bytes, repair_report: Dict[str, Any], rebuilt_path: str) -> List[Dict[str, Any]]:
    exact = sha256_bytes(raw_bytes or b"") == sha256_bytes(decoded_raw or b"")
    if ACTIVE_ECC_ENABLED:
        decode_success = bool(repair_report.get("repair_success")) and exact
    else:
        decode_success = exact
    return [
        {"Property": "Decode successful", "Value": "Yes" if decode_success else "No"},
        {"Property": "Raw data exactly recovered", "Value": "Yes" if exact else "No"},
        {"Property": "Output rebuilt", "Value": "Yes" if rebuilt_path else "No"},
        {"Property": "Conclusion", "Value": "Recovered raw representation can be rebuilt and matches the source." if exact else "Decoded output was rebuilt, but raw data still contains differences."},
    ]

# -----------------------------------------------------------------------------
# Panels
# -----------------------------------------------------------------------------


def render_panel_1_input() -> None:
    with st.container(border=True):
        step_header(1, PANEL_TITLES["input"])
        left, right = st.columns(2, gap="large")
        with left:
            uploaded = st.file_uploader("", type=None, key=_key("upload_input"))
            if uploaded is not None:
                _store_upload(uploaded)
        with right:
            data = st.session_state.get(_key("input_bytes"), b"")
            path = st.session_state.get(_key("input_path"), "")
            name = st.session_state.get(_key("input_name"), "")
            if not data or not path:
                st.info("Upload an image, UTF-8 text file, or audio file to start.")
                return
            domain = detect_domain(name, data)
            _metrics_row([("Detected domain", domain), ("Input size", fmt_bytes(len(data))), ("SHA256", sha256_bytes(data)[:12] + "...")])
            preview_file_streamlit(st, path, "Original preview", key_suffix="input")
            with st.expander("Original binary", expanded=False):
                bit_text = bytes_to_bitstring(data)
                st.text_area("Binary bitstream", bit_text[:3000] + ("..." if len(bit_text) > 3000 else ""), height=120, key=_content_key("input_bits_preview", bit_text))
                _download_text_button("Download input binary", bit_text, "input_binary.txt", key=_key("download_input_binary"))


def render_panel_2_raw_representation() -> None:
    with st.container(border=True):
        step_header(2, PANEL_TITLES["raw"])
        data = st.session_state.get(_key("input_bytes"), b"")
        path = st.session_state.get(_key("input_path"), "")
        name = st.session_state.get(_key("input_name"), "")
        if not data or not path:
            st.info("Upload a file first.")
            return
        domain = detect_domain(name, data)
        if domain not in {"image", "text", "audio"}:
            st.error("This first version supports image, UTF-8 text, and audio waveform only.")
            return
        st.caption("This panel removes the fragile container layer and creates a raw/canonical data representation before binary and DNA encoding.")
        options: Dict[str, Any] = {}
        if domain == "image":
            a, b, c = st.columns(3)
            options["image_mode"] = a.selectbox("Image raw mode", ["RGB pixels", "Grayscale pixels", "Binary pixels"], index=0, key=_key("image_mode"))
            options["threshold"] = b.slider("Binary threshold", 0, 255, 128, key=_key("image_threshold"), disabled=options["image_mode"] != "Binary pixels")
            options["max_side"] = c.selectbox("Resize max side", [0, 512, 256, 128, 64], index=0, format_func=lambda x: "Original" if x == 0 else f"{x} px", key=_key("image_max_side"))
        elif domain == "text":
            options["normalize_line_endings"] = st.checkbox("Normalize line endings to LF", value=True, key=_key("text_normalize"))
            st.caption("Text is stored as UTF-8 bytes. SHA256 is only used for integrity checking, not as recoverable data.")
        elif domain == "audio":
            a, b = st.columns(2)
            options["sample_rate"] = a.selectbox("Waveform sample rate", ["Original", "44100", "22050", "16000", "8000"], index=0, key=_key("audio_sample_rate"))
            options["channels"] = b.selectbox("Channels", ["Original", "Mono", "Stereo"], index=0, key=_key("audio_channels"))
            st.caption("Audio is converted to PCM16 waveform bytes and rebuilt as WAV. Non-WAV formats need FFmpeg on PATH.")

        option_sig = json.dumps({"domain": domain, **options}, sort_keys=True)
        current_sig = st.session_state.get(_key("raw_signature"))
        if st.button("Run Raw Representation", key=_key("run_raw")) or (st.session_state.get(_key("raw_bytes")) and current_sig != option_sig):
            _clear_downstream("raw")
            try:
                raw_bytes, raw_meta, preview_path = build_raw_representation(path, data, domain=domain, options=options)
            except Exception as exc:
                st.error(str(exc))
                return
            st.session_state[_key("raw_bytes")] = raw_bytes
            st.session_state[_key("raw_meta")] = raw_meta
            st.session_state[_key("raw_preview_path")] = preview_path
            st.session_state[_key("raw_signature")] = option_sig
            if current_sig and current_sig != option_sig:
                st.caption("Raw representation was updated automatically because settings changed.")

        raw_bytes = st.session_state.get(_key("raw_bytes"), b"")
        raw_meta = st.session_state.get(_key("raw_meta"), {}) or {}
        preview_path = st.session_state.get(_key("raw_preview_path"), "")
        if not raw_bytes:
            st.info("Run Raw Representation to continue.")
            return
        _metrics_row([("Raw domain", raw_meta.get("domain", "—")), ("Representation", raw_meta.get("representation", "—")), ("Raw size", fmt_bytes(len(raw_bytes))), ("Expansion vs container", f"{len(raw_bytes) / max(1, len(data)):.2f}×")])
        st.dataframe(pd.DataFrame(raw_meta_summary(raw_meta)), use_container_width=True, hide_index=True)
        preview_file_streamlit(st, preview_path, "Raw representation preview", key_suffix="raw")
        d1, d2 = st.columns(2)
        with d1:
            _download_bytes_button("Download raw bytes", raw_bytes, "raw_representation.bin", key=_key("download_raw_bytes"))
        with d2:
            _download_text_button("Download raw metadata", json.dumps(raw_meta, indent=2), "raw_metadata.json", key=_key("download_raw_meta"))
        # raw_bytes = st.session_state.get(_key("raw_bytes"), b"") or b""

        # if raw_bytes:
        #     raw_bits = bytes_to_bitstring(raw_bytes)
        
        #     st.markdown("#### Prepared payload preview")
        #     st.caption("Binary payload generated from the raw representation and passed to the SM/R∞ DNA design step.")
        
        #     st.text_area(
        #         "Prepared binary payload",
        #         raw_bits[:5000] + ("..." if len(raw_bits) > 5000 else ""),
        #         height=220,
        #         key=_content_key("raw_binary_preview", raw_bits),
        #     )
        # else:
        #     st.info("Run Raw Representation first.")
        # stored_bits = bytes_to_bitstring(raw_bytes)
        
        # st.markdown("#### Prepared payload preview")
        # st.caption("Binary payload that will be passed to the SM/R∞ DNA design step.")
        
        # st.text_area(
        #     "Prepared binary payload",
        #     stored_bits[:5000] + ("..." if len(stored_bits) > 5000 else ""),
        #     height=220,
        #     key=_content_key(prefix, "stored_binary_preview", stored_bits),
        # )
        
        # d1, d2 = st.columns(2)
        
        # with d1:
        #     _download_bytes_button(
        #         BUTTONS["download_stored_data"],
        #         stored,
        #         f"stored_data{md.get('ext', '.bin')}",
        #         key=_key(prefix, "download_stored_data"),
        #     )
        
        # with d2:
        #     _download_text_button(
        #         BUTTONS["download_stored_binary"],
        #         stored_bits,
        #         "stored_binary.txt",
        #         key=_key(prefix, "download_stored_binary"),
        #     )

def render_panel_3_dna_encoding() -> None:
    with st.container(border=True):
        step_header(3, PANEL_TITLES["dna"])
        raw_bytes = st.session_state.get(_key("raw_bytes"), b"")
        if not raw_bytes:
            st.info("Run Raw Representation first.")
            return

        mapping = st.selectbox("DNA design rule", MAPPING_OPTIONS, index=0, format_func=display_mapping, key=_key("mapping"))
        ecc_label = "Reed-Solomon" if ACTIVE_ECC_ENABLED else "None"
        ecc_meta: Dict[str, Any] = {"ecc": "None", "original_size": len(raw_bytes)}
        data_block_size = 64
        parity_bytes = 64

        if ACTIVE_ECC_ENABLED:
            st.markdown("#### ECC option")
            a, b = st.columns(2)
            data_block_size = int(a.number_input("RS data block bytes", min_value=8, max_value=223, value=64, step=8, key=_key("rs_data_block")))
            max_parity = max(2, 255 - data_block_size)
            default_parity = min(64, max_parity)
            parity_bytes = int(b.number_input("RS parity bytes", min_value=2, max_value=max_parity, value=default_parity, step=2, key=_key("rs_parity")))
            st.caption("Reed-Solomon protects raw bytes before SM/R∞ DNA encoding. Keep substitution-only for the cleanest ECC test.")
        else:
            st.caption("ECC option: None. This tab shows raw-representation recovery without error correction.")

        encode_sig = json.dumps({
            "raw_sha": sha256_bytes(raw_bytes),
            "mapping": mapping,
            "ecc": ecc_label,
            "data_block_size": data_block_size if ACTIVE_ECC_ENABLED else 0,
            "parity_bytes": parity_bytes if ACTIVE_ECC_ENABLED else 0,
        }, sort_keys=True)

        if st.button("Run DNA Encoding", key=_key("run_dna")) or (st.session_state.get(_key("dna")) and st.session_state.get(_key("encode_signature")) != encode_sig):
            _clear_downstream("dna")
            try:
                if ACTIVE_ECC_ENABLED:
                    rs_result = encode_rs_bytes(raw_bytes, data_block_size=data_block_size, parity_bytes=parity_bytes)
                    encoding_bytes = rs_result.protected_bytes
                    ecc_meta = dict(rs_result.meta)
                else:
                    encoding_bytes = raw_bytes
                    ecc_meta = {"ecc": "None", "original_size": len(raw_bytes), "protected_size": len(raw_bytes), "ecc_overhead_ratio": 1.0}
                dna, bits, meta = encode_bytes_to_dna(encoding_bytes, mapping)
            except Exception as exc:
                st.error(str(exc))
                return
            meta.update({
                "dna_design_rule": mapping,
                "ecc_option": ecc_label,
                "raw_bytes_len": len(raw_bytes),
                "encoding_bytes_len": len(encoding_bytes),
            })
            st.session_state[_key("encoding_bytes")] = encoding_bytes
            st.session_state[_key("ecc_meta")] = ecc_meta
            st.session_state[_key("dna")] = dna
            st.session_state[_key("bits")] = bits
            st.session_state[_key("codec_meta")] = meta
            st.session_state[_key("encode_signature")] = encode_sig

        dna = st.session_state.get(_key("dna"), "")
        bits = st.session_state.get(_key("bits"), "")
        encoding_bytes = st.session_state.get(_key("encoding_bytes"), b"")
        ecc_meta = st.session_state.get(_key("ecc_meta"), {}) or {}
        if not dna:
            st.info("Run DNA Encoding to continue.")
            return
        hp = homopolymer_stats(dna)
        _metrics_row([
            ("DNA design rule", display_mapping(mapping)),
            ("ECC option", ecc_label),
            ("Raw bytes", fmt_bytes(len(raw_bytes))),
            ("Encoded bytes", fmt_bytes(len(encoding_bytes))),
            ("DNA length", f"{len(dna):,} nt"),
        ])
        _metrics_row([
            ("DNA expansion", f"{len(dna) / max(1, len(raw_bytes) * 4):.2f}× vs raw-SM baseline"),
            ("ECC overhead", f"{float(ecc_meta.get('ecc_overhead_ratio', 1.0)):.2f}×" if ACTIVE_ECC_ENABLED else "1.00×"),
            ("GC content", f"{gc_content(dna):.3f}"),
            ("Longest HP", hp.get("longest", 0)),
            ("Homopolymer segments ≥2", hp.get("count_ge2", 0)),
        ])
        if ACTIVE_ECC_ENABLED:
            st.dataframe(pd.DataFrame([
                {"Property": "RS data block bytes", "Value": ecc_meta.get("data_block_size", "—")},
                {"Property": "RS parity bytes", "Value": ecc_meta.get("parity_bytes", "—")},
                {"Property": "Codeword size", "Value": ecc_meta.get("codeword_size", "—")},
                {"Property": "RS blocks", "Value": ecc_meta.get("blocks", "—")},
                {"Property": "Max unknown byte errors/block", "Value": ecc_meta.get("max_unknown_byte_errors_per_block", "—")},
            ]), use_container_width=True, hide_index=True)
        st.text_area("Base string", _preview_seq(dna, 900), height=150, key=_content_key("dna_preview", dna))
        d1, d2 = st.columns(2)
        with d1:
            _download_text_button("Download encoded DNA", dna, "encoded_dna.txt", key=_key("download_encoded_dna"))
        with d2:
            _download_text_button("Download encoded binary", bits, "encoded_binary.txt", key=_key("download_encoded_binary"))


def render_panel_4_strand_errors() -> None:
    with st.container(border=True):
        step_header(4, PANEL_TITLES["strand"])
        dna = st.session_state.get(_key("dna"), "")
        if not dna:
            st.info("Run DNA Encoding first.")
            return
        st.markdown("#### Strand Preparation")
        with st.expander("Strand design settings", expanded=not bool(st.session_state.get(_key("strand_rows")))):
            a, b = st.columns(2)
            total_len = a.number_input("Total strand length", min_value=80, max_value=250, value=125, step=1, key=_key("strand_total_len"))
            index_len = b.number_input("Index length", min_value=0, max_value=24, value=12, step=1, key=_key("strand_index_len"))
            fbr = st.text_input("FBR", value=DEFAULT_FBR, key=_key("fbr"))
            rbr = st.text_input("RBR", value=DEFAULT_RBR, key=_key("rbr"))
            build_clicked = st.button("Run Strand Preparation", key=_key("run_strand"))
        strand_sig = f"{hashlib.sha256(clean_dna(dna).encode()).hexdigest()}|{int(total_len)}|{int(index_len)}|{clean_dna(fbr)}|{clean_dna(rbr)}"
        if build_clicked or st.session_state.get(_key("strand_signature")) != strand_sig:
            try:
                rows = _make_prepared_strands(dna, total_len=int(total_len), index_len=int(index_len), fbr=fbr, rbr=rbr)
            except Exception as exc:
                st.error(str(exc))
                return
            st.session_state[_key("strand_rows")] = rows
            st.session_state[_key("strand_signature")] = strand_sig
            for k in ["noisy_strand_rows", "noisy_dna", "error_events", "error_metrics", "decoded_payload_bytes", "decoded_raw_bytes", "ecc_repair_report", "rebuilt_path"]:
                st.session_state.pop(_key(k), None)
        rows: List[Dict[str, Any]] = st.session_state.get(_key("strand_rows"), []) or []
        if not rows:
            st.info("Run Strand Preparation to continue.")
            return
        total_full_len = sum(len(clean_dna(r.get("Full strand", ""))) for r in rows)
        _metrics_row([("Designed strands", len(rows)), ("Total strand length", f"{total_full_len:,} nt"), ("Strand Design length increase", f"{total_full_len / max(1, len(clean_dna(dna))):.2f}×"), ("DNA design rule", display_mapping(st.session_state.get(_key("mapping"), "")))])
        st.dataframe(_strand_summary(rows), use_container_width=True, hide_index=True)
        selected = st.selectbox("Inspect designed strand", [str(r["No."]) for r in rows], key=_key("inspect_strand"))
        row = rows[int(selected) - 1]
        _render_segmented_strand(row, "Designed strand")

        st.markdown("---")
        st.markdown("#### Add DNA errors")
        a, b, c, d = st.columns(4)
        scope = a.selectbox("Error target", ["Payload only", "Full strand"], index=0, key=_key("error_scope"))
        sub = b.number_input("Substitution", min_value=0.0, max_value=0.2, value=0.001, step=0.001, format="%.4f", key=_key("sub_rate"))
        seed = c.number_input("Seed", min_value=1, max_value=999999, value=7, step=1, key=_key("error_seed"))
        allow_indels = d.checkbox("Allow indels", value=False, key=_key("allow_indels"))
        e, f = st.columns(2)
        ins = e.number_input("Insertion", min_value=0.0, max_value=0.2, value=0.0, step=0.001, format="%.4f", key=_key("ins_rate"), disabled=not allow_indels)
        dele = f.number_input("Deletion", min_value=0.0, max_value=0.2, value=0.0, step=0.001, format="%.4f", key=_key("del_rate"), disabled=not allow_indels)
        if allow_indels:
            st.warning("Insertion/deletion changes DNA length and can break SM/R∞ framing. Use substitution-only first.")
        if st.button("Run Add Errors", key=_key("run_errors")):
            for k in ["decoded_payload_bytes", "decoded_raw_bytes", "decoded_bits", "decoded_meta", "ecc_repair_report", "rebuilt_path", "decode_note"]:
                st.session_state.pop(_key(k), None)
            err_rows, noisy_dna, events, metrics = _mutate_prepared_rows(rows, scope=scope, substitution_rate=float(sub), insertion_rate=float(ins), deletion_rate=float(dele), seed=int(seed), allow_indels=bool(allow_indels))
            st.session_state[_key("noisy_strand_rows")] = err_rows
            st.session_state[_key("noisy_dna")] = noisy_dna
            st.session_state[_key("error_events")] = events
            st.session_state[_key("error_metrics")] = metrics
        noisy_dna = st.session_state.get(_key("noisy_dna"), "")
        if not noisy_dna:
            st.info("Run Add Errors to continue.")
            return
        metrics = st.session_state.get(_key("error_metrics"), {}) or {}
        _metrics_row([("Noisy DNA length", f"{len(clean_dna(noisy_dna)):,} nt"), ("Substitutions", metrics.get("substitutions", 0)), ("Insertions", metrics.get("insertions", 0)), ("Deletions", metrics.get("deletions", 0)), ("Total errors", metrics.get("total_errors", 0))])
        st.text_area("Noisy DNA preview", _preview_seq(noisy_dna, 900), height=120, key=_key("noisy_preview"))
        events = st.session_state.get(_key("error_events"), []) or []
        if events:
            st.dataframe(pd.DataFrame(events[:1000]), use_container_width=True, hide_index=True)
        err_rows = st.session_state.get(_key("noisy_strand_rows"), []) or []
        if err_rows:
            erow = err_rows[int(selected) - 1]
            ev_pos = {int(ev.get("Full-strand position")) for ev in events if str(ev.get("Strand")) == selected and str(ev.get("Full-strand position", "")).isdigit() and ev.get("Operation") in {"substitution", "deletion"}}
            _render_segmented_strand(erow, "Error strand", error_positions=ev_pos)
        _download_text_button("Download noisy DNA", noisy_dna, "noisy_dna.txt", key=_key("download_noisy_dna"))


def render_panel_5_decode_rebuild() -> None:
    with st.container(border=True):
        step_header(5, PANEL_TITLES["decode"])
        noisy_dna = st.session_state.get(_key("noisy_dna"), "")
        if not noisy_dna:
            st.info("Run Add Errors first.")
            return

        raw_meta = st.session_state.get(_key("raw_meta"), {}) or {}
        mapping = st.session_state.get(_key("mapping"), "Simple Mapping")
        ecc_meta = st.session_state.get(_key("ecc_meta"), {}) or {}

        if st.button("Run Decode", key=_key("run_decode")):
            try:
                decoded_payload, bits, dec_meta = decode_dna_to_bytes(noisy_dna, mapping)
                if ACTIVE_ECC_ENABLED:
                    decoded_raw, repair_report = decode_rs_bytes(decoded_payload, ecc_meta)
                    note = "Reed-Solomon repair completed." if repair_report.get("repair_success") else "Reed-Solomon repair failed for one or more blocks."
                else:
                    decoded_raw = decoded_payload
                    repair_report = {
                        "repair_success": None,
                        "blocks": 0,
                        "corrected_blocks": 0,
                        "failed_blocks": 0,
                        "corrected_symbols": 0,
                        "block_reports": [],
                    }
                    note = "No ECC repair was applied. Decoded DNA bytes are used directly as raw data."

                expected = int(raw_meta.get("expected_raw_bytes", len(decoded_raw)))
                if len(decoded_raw) < expected:
                    note += f" Rebuild pads {expected - len(decoded_raw)} missing raw bytes."
                elif len(decoded_raw) > expected:
                    note += f" Rebuild truncates {len(decoded_raw) - expected} extra raw bytes."

                rebuilt_path = rebuild_raw_to_file(
                    decoded_raw,
                    raw_meta,
                    preferred_name=("decoded_raw_output_ecc" if ACTIVE_ECC_ENABLED else "decoded_raw_output_noecc"),
                )
            except Exception as exc:
                st.error(str(exc))
                return

            st.session_state[_key("decoded_payload_bytes")] = decoded_payload
            st.session_state[_key("decoded_raw_bytes")] = decoded_raw
            st.session_state[_key("decoded_bits")] = bits
            st.session_state[_key("decoded_meta")] = dec_meta
            st.session_state[_key("ecc_repair_report")] = repair_report
            st.session_state[_key("rebuilt_path")] = rebuilt_path
            st.session_state[_key("decode_note")] = note

        decoded_raw = st.session_state.get(_key("decoded_raw_bytes"), None)
        decoded_payload = st.session_state.get(_key("decoded_payload_bytes"), b"")
        rebuilt_path = st.session_state.get(_key("rebuilt_path"), "")
        if decoded_raw is None or not rebuilt_path:
            st.info("Run Decode to rebuild and inspect the recovered raw output.")
            return

        raw_bytes = st.session_state.get(_key("raw_bytes"), b"")
        encoding_bytes = st.session_state.get(_key("encoding_bytes"), raw_bytes)
        repair_report = st.session_state.get(_key("ecc_repair_report"), {}) or {}
        exact_match = sha256_bytes(raw_bytes or b"") == sha256_bytes(decoded_raw or b"")

        if ACTIVE_ECC_ENABLED:
            rs_status = "Yes" if repair_report.get("repair_success") else "No"
        else:
            rs_status = "Not used"

        _metrics_row([
            ("Decode output", "Rebuilt" if rebuilt_path else "Failed"),
            ("Raw byte accuracy", f"{byte_accuracy(raw_bytes, decoded_raw):.6f}"),
            ("SHA256 match", "Yes" if exact_match else "No"),
            ("RS repair", rs_status),
        ])
        st.caption(st.session_state.get(_key("decode_note"), ""))

        if str(raw_meta.get("domain", "")) == "text":
            original_text = bytes_to_preview_text(raw_bytes, 20000)
            decoded_text = bytes_to_preview_text(decoded_raw, 20000)
            st.markdown("#### Decode / Repair Summary")
            st.dataframe(pd.DataFrame([
                {"Property": "Source", "Value": "Noisy encoded DNA" if st.session_state.get(_key("error_metrics")) else "Current encoded DNA"},
                {"Property": "DNA design rule", "Value": display_mapping(mapping)},
                {"Property": "ECC option", "Value": "Reed-Solomon" if ACTIVE_ECC_ENABLED else "None"},
                {"Property": "Decode status", "Value": "Pass" if decoded_text else "Review"},
                {"Property": "RS repair", "Value": rs_status},
                {"Property": "Output status", "Value": "Readable text" if decoded_text else "Failed"},
            ]), use_container_width=True, hide_index=True)

            st.markdown("#### Decoded Text Preview")
            p1, p2 = st.columns(2, gap="large")
            with p1:
                st.markdown("##### Original text")
                st.text_area("Original text preview", original_text[:5000], height=240, key=_key("step5_text_original_preview"))
            with p2:
                st.markdown("##### Final recovered text" + (" / RS repaired" if ACTIVE_ECC_ENABLED else ""))
                st.text_area("Final recovered text preview", decoded_text[:5000], height=240, key=_key("step5_text_final_preview"))

            d1, d2 = st.columns(2)
            with d1:
                _download_text_button("Download recovered text", decoded_text, "decoded_recovered_text.txt", key=_key("download_decoded_text"))
            with d2:
                _download_bytes_button("Download recovered raw bytes", decoded_raw, "decoded_recovered_raw_bytes.bin", key=_key("download_decoded_raw"))
        else:
            preview_file_streamlit(st, rebuilt_path, "Decoded preview", key_suffix=_key("decoded"))

            d1, d2 = st.columns(2)
            with d1:
                _download_bytes_button("Download recovered raw bytes", decoded_raw, "decoded_recovered_raw_bytes.bin", key=_key("download_decoded_raw"))
            with d2:
                _download_bytes_button("Download rebuilt output", Path(rebuilt_path).read_bytes(), f"rebuilt_output{Path(rebuilt_path).suffix}", key=_key("download_rebuilt"))

        if ACTIVE_ECC_ENABLED and repair_report.get("block_reports"):
            with st.expander("Reed-Solomon block repair details", expanded=False):
                st.dataframe(pd.DataFrame(repair_report.get("block_reports", [])), use_container_width=True, hide_index=True)


def render_panel_6_summary() -> None:
    with st.container(border=True):
        step_header(6, PANEL_TITLES["summary"])
        if not st.session_state.get(_key("rebuilt_path")):
            st.info("Run Decode first.")
            return

        input_path = st.session_state.get(_key("input_path"), "")
        input_bytes = st.session_state.get(_key("input_bytes"), b"") or b""
        raw_bytes = st.session_state.get(_key("raw_bytes"), b"") or b""
        decoded_raw = st.session_state.get(_key("decoded_raw_bytes"), b"") or b""
        raw_meta = st.session_state.get(_key("raw_meta"), {}) or {}
        raw_preview_path = st.session_state.get(_key("raw_preview_path"), "")
        rebuilt_path = st.session_state.get(_key("rebuilt_path"), "")
        dna = st.session_state.get(_key("dna"), "")
        noisy = st.session_state.get(_key("noisy_dna"), "")
        strand_rows = st.session_state.get(_key("strand_rows"), []) or []
        errors = st.session_state.get(_key("error_metrics"), {}) or {}
        mapping = st.session_state.get(_key("mapping"), "")
        encoding_bytes = st.session_state.get(_key("encoding_bytes"), raw_bytes)
        decoded_payload = st.session_state.get(_key("decoded_payload_bytes"), b"") or b""
        ecc_meta = st.session_state.get(_key("ecc_meta"), {}) or {}
        repair_report = st.session_state.get(_key("ecc_repair_report"), {}) or {}

        st.markdown("#### 📊 Summary")
        original_col, encoded_col, decoded_col = st.columns(3, gap="large")

        if str(raw_meta.get("domain", "")) == "text":
            original_text = bytes_to_preview_text(raw_bytes, 20000)
            decoded_text = bytes_to_preview_text(decoded_raw, 20000)
            method_preview = dna or ""

            with original_col:
                st.markdown("##### Original")
                st.text_area("Original preview", original_text[:5000], height=220, key=_key("summary_text_original_preview"))
                st.dataframe(pd.DataFrame([
                    {"Property": "Characters", "Value": f"{len(original_text):,}"},
                    {"Property": "Words", "Value": f"{len(original_text.split()):,}"},
                    {"Property": "Raw size", "Value": fmt_bytes(len(raw_bytes or b""))},
                ]), use_container_width=True, hide_index=True)

            with encoded_col:
                st.markdown("##### Raw / Encoded")
                st.text_area("Encoded DNA preview", _preview_seq(method_preview, 900), height=220, key=_key("summary_text_encoded_preview"))
                st.dataframe(pd.DataFrame([
                    {"Property": "Raw representation", "Value": raw_meta.get("representation", "UTF-8 text bytes")},
                    {"Property": "DNA design rule", "Value": display_mapping(mapping)},
                    {"Property": "ECC option", "Value": "Reed-Solomon" if ACTIVE_ECC_ENABLED else "None"},
                    {"Property": "DNA length", "Value": f"{len(clean_dna(dna)):,} nt" if dna else "—"},
                ]), use_container_width=True, hide_index=True)

            with decoded_col:
                st.markdown("##### Decoded")
                st.text_area("Decoded preview", decoded_text[:5000], height=220, key=_key("summary_text_decoded_preview"))
                st.dataframe(pd.DataFrame([
                    {"Property": "Decoded length", "Value": f"{len(decoded_text):,} chars"},
                    {"Property": "Output status", "Value": "Readable" if decoded_text else "Failed"},
                    {"Property": "Raw byte accuracy", "Value": f"{byte_accuracy(raw_bytes, decoded_raw):.6f}"},
                    {"Property": "SHA256 raw match", "Value": "Yes" if sha256_bytes(raw_bytes or b"") == sha256_bytes(decoded_raw or b"") else "No"},
                ]), use_container_width=True, hide_index=True)
        else:
            with original_col:
                st.markdown("##### Original")
                if input_path and input_bytes:
                    preview_file_streamlit(st, input_path, "Original preview", key_suffix=_key("summary_original"))
                    _render_property_table(_basic_file_rows(input_path, input_bytes, label="Original container"))
                else:
                    st.info("Upload a file first.")

            with encoded_col:
                st.markdown("##### Raw / Encoded")
                if raw_preview_path and raw_bytes:
                    preview_file_streamlit(st, raw_preview_path, "Raw representation preview", key_suffix=_key("summary_raw"))
                    _render_property_table(_raw_encoded_rows(raw_meta, raw_bytes, encoding_bytes, dna, mapping, ecc_meta))
                else:
                    st.info("Run Raw Representation and Encoding first.")

            with decoded_col:
                st.markdown("##### Decoded")
                if rebuilt_path:
                    preview_file_streamlit(st, rebuilt_path, "Decoded preview", key_suffix=_key("summary_decoded"))
                    _render_property_table(_decoded_result_rows(raw_bytes, decoded_raw, raw_meta, repair_report, rebuilt_path))
                else:
                    st.info("Run Decode first.")

        st.markdown("#### 🧬 Encoding statistics")
        _render_property_table(_encoding_statistics_rows(raw_meta, raw_bytes, encoding_bytes, dna, strand_rows, mapping, ecc_meta))

        st.markdown("#### ⚠️ Error Adding Report")
        _render_property_table(_error_report_rows(dna, noisy, errors))

        st.markdown("#### 🔁 Decode / Recovery Report")
        _render_property_table(_decode_recovery_rows(raw_bytes, encoding_bytes, decoded_payload, decoded_raw, repair_report))

        st.markdown("#### ✅ Recovery Quality Report")
        quality_rows = _select_key_quality_rows(raw_bytes, decoded_raw, raw_meta)
        st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 🧾 Final summary")
        _render_property_table(_final_summary_rows(raw_bytes, decoded_raw, repair_report, rebuilt_path))

        summary_rows = (
            _encoding_statistics_rows(raw_meta, raw_bytes, encoding_bytes, dna, strand_rows, mapping, ecc_meta)
            + _error_report_rows(dna, noisy, errors)
            + _decode_recovery_rows(raw_bytes, encoding_bytes, decoded_payload, decoded_raw, repair_report)
            + quality_rows
            + _final_summary_rows(raw_bytes, decoded_raw, repair_report, rebuilt_path)
        )
        buf = io.StringIO()
        pd.DataFrame(summary_rows).to_csv(buf, index=False)
        _download_text_button("Download summary CSV", buf.getvalue(), "raw_representation_summary.csv", key=_key("download_summary"))


def render_pipeline(prefix: str, ecc_enabled: bool) -> None:
    _set_pipeline_context(prefix, ecc_enabled)
    render_stepper()
    render_panel_1_input()
    render_panel_2_raw_representation()
    render_panel_3_dna_encoding()
    render_panel_4_strand_errors()
    render_panel_5_decode_rebuild()
    render_panel_6_summary()


def render_app_body() -> None:
    st.markdown(
        """
<div class="hero-card">
  <div class="hero-title"><br/>🧬 DNA Error Simulation and Reed–Solomon ECC Recovery Pipeline</div>
   <div class="hero-subtitle">Mode: Raw Representation Storage</div>
</div>
""",
        unsafe_allow_html=True,
    )
    tab1, tab2 = st.tabs(["No ECC — Raw Representation", "RS-ECC Recovery — Raw Representation"])
    with tab1:
        render_pipeline(prefix="noecc", ecc_enabled=False)
    with tab2:
        render_pipeline(prefix="ecc", ecc_enabled=True)
