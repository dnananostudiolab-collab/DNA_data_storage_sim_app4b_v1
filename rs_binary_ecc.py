from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from gf_rs_generic import GF2m

GF256 = GF2m(8, 0x11D)


@dataclass
class RSEncodeResult:
    protected_bytes: bytes
    meta: Dict[str, Any]


def encode_rs_bytes(data: bytes, *, data_block_size: int = 64, parity_bytes: int = 32) -> RSEncodeResult:
    raw = bytes(data or b"")
    k = int(data_block_size)
    nsym = int(parity_bytes)
    if k <= 0:
        raise ValueError("data_block_size must be positive")
    if nsym <= 0:
        raise ValueError("parity_bytes must be positive")
    if k + nsym > 255:
        raise ValueError("For GF(2^8), data_block_size + parity_bytes must be <= 255")

    blocks = max(1, math.ceil(len(raw) / k))
    out = bytearray()
    for block_id in range(blocks):
        block = raw[block_id * k:(block_id + 1) * k]
        if len(block) < k:
            block = block + bytes(k - len(block))
        code = GF256.rs_encode_msg(list(block), nsym)
        out.extend(bytes(code))

    meta = {
        "ecc": "Reed-Solomon byte-level ECC",
        "field": "GF(2^8)",
        "primitive_poly": "0x11D",
        "original_size": len(raw),
        "data_block_size": k,
        "parity_bytes": nsym,
        "codeword_size": k + nsym,
        "blocks": blocks,
        "protected_size": len(out),
        "ecc_overhead_bytes": len(out) - len(raw),
        "ecc_overhead_ratio": (len(out) / len(raw)) if raw else 1.0,
        "max_unknown_byte_errors_per_block": nsym // 2,
    }
    return RSEncodeResult(protected_bytes=bytes(out), meta=meta)


def decode_rs_bytes(protected: bytes, meta: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    protected = bytes(protected or b"")
    k = int(meta.get("data_block_size", 64))
    nsym = int(meta.get("parity_bytes", 32))
    n = int(meta.get("codeword_size", k + nsym))
    blocks = int(meta.get("blocks", max(1, math.ceil(len(protected) / max(1, n)))))
    original_size = int(meta.get("original_size", -1))

    recovered = bytearray()
    failed_blocks = 0
    corrected_blocks = 0
    corrected_symbols_total = 0
    block_reports: List[Dict[str, Any]] = []

    for block_id in range(blocks):
        chunk = protected[block_id * n:(block_id + 1) * n]
        length_note = "exact"
        if len(chunk) < n:
            chunk = chunk + bytes(n - len(chunk))
            length_note = "padded_short_codeword"
        elif len(chunk) > n:
            chunk = chunk[:n]
            length_note = "truncated_long_codeword"

        try:
            msg = GF256.rs_correct_msg(list(chunk), nsym)
            reencoded = bytes(GF256.rs_encode_msg(list(msg), nsym))
            corrected_symbols = sum(1 for a, b in zip(bytes(chunk), reencoded) if a != b)
            corrected_symbols_total += corrected_symbols
            if corrected_symbols > 0:
                corrected_blocks += 1
            recovered.extend(bytes(msg[:k]))
            block_reports.append({
                "Block": block_id + 1,
                "Status": "corrected" if corrected_symbols else "clean",
                "Corrected symbols": corrected_symbols,
                "Note": length_note,
            })
        except Exception as exc:
            failed_blocks += 1
            # Fallback: keep systematic data part so the user can still compare how bad it is.
            recovered.extend(bytes(chunk[:k]))
            block_reports.append({
                "Block": block_id + 1,
                "Status": "failed",
                "Corrected symbols": 0,
                "Note": f"{length_note}; {exc}",
            })

    if original_size >= 0:
        recovered_bytes = bytes(recovered[:original_size])
    else:
        recovered_bytes = bytes(recovered)

    report: Dict[str, Any] = {
        "repair_success": failed_blocks == 0,
        "blocks": blocks,
        "corrected_blocks": corrected_blocks,
        "failed_blocks": failed_blocks,
        "corrected_symbols": corrected_symbols_total,
        "output_size": len(recovered_bytes),
        "block_reports": block_reports,
    }
    return recovered_bytes, report
