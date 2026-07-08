from __future__ import annotations

from typing import Any, Dict, List, Tuple

from utils_core import bitstring_to_bytes, bytes_to_bitstring

BASES = "ACGT"
BITS2BASE = {"00": "A", "01": "C", "10": "G", "11": "T"}
BASE2BITS = {v: k for k, v in BITS2BASE.items()}
DIMERS = [a + b for a in BASES for b in BASES]
DIMER2VAL = {d: i for i, d in enumerate(DIMERS)}
VAL2DIMER = {i: d for i, d in enumerate(DIMERS)}

MAPPING_OPTIONS = ["Simple Mapping", "RINF_B16"]
MAPPING_DISPLAY = {"Simple Mapping": "SM", "RINF_B16": "R∞"}


def clean_dna(seq: str) -> str:
    return "".join(ch for ch in str(seq or "").upper() if ch in BASES)


def gc_content(seq: str) -> float:
    seq = clean_dna(seq)
    return (sum(ch in "GC" for ch in seq) / len(seq)) if seq else 0.0


def homopolymer_stats(seq: str) -> Dict[str, int]:
    seq = clean_dna(seq)
    if not seq:
        return {"longest": 0, "count_ge2": 0, "total_runs": 0}
    runs: List[int] = []
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return {
        "longest": max(runs),
        "count_ge2": sum(1 for r in runs if r >= 2),
        "count_ge3": sum(1 for r in runs if r >= 3),
        "count_ge4": sum(1 for r in runs if r >= 4),
        "total_runs": len(runs),
    }


def display_mapping(mapping: str) -> str:
    return MAPPING_DISPLAY.get(mapping, mapping)


def encode_bytes_to_dna(data: bytes, mapping: str) -> Tuple[str, str, Dict[str, Any]]:
    data = bytes(data or b"")
    bits = bytes_to_bitstring(data)
    if mapping == "Simple Mapping":
        # Header base stores the padding used to complete the final 2-bit group.
        pad = (2 - (len(bits) % 2)) % 2
        padded = bits + ("0" * pad)
        header = "A" if pad == 0 else "C"
        dna = header + "".join(BITS2BASE[padded[i:i + 2]] for i in range(0, len(padded), 2))
        meta = {
            "mapping": mapping,
            "display": display_mapping(mapping),
            "mode": "SM_2BIT",
            "bytes_len": len(data),
            "bits_len": len(bits),
            "pad_bits": pad,
            "dna_len": len(dna),
        }
        return dna, bits, meta

    if mapping == "RINF_B16":
        # R∞ allows all 16 dimers.  Here each byte is represented by two hex-like dimers.
        out: List[str] = []
        for b in data:
            out.append(VAL2DIMER[(b >> 4) & 0xF])
            out.append(VAL2DIMER[b & 0xF])
        dna = "".join(out)
        meta = {
            "mapping": mapping,
            "display": display_mapping(mapping),
            "mode": "RINF_DIRECT_DIMER",
            "bytes_len": len(data),
            "bits_len": len(bits),
            "dna_len": len(dna),
            "dimers": len(dna) // 2,
        }
        return dna, bits, meta

    raise ValueError(f"Unsupported mapping: {mapping}")


def decode_dna_to_bytes(dna: str, mapping: str) -> Tuple[bytes, str, Dict[str, Any]]:
    dna = clean_dna(dna)
    if mapping == "Simple Mapping":
        if not dna:
            return b"", "", {"mapping": mapping, "mode": "SM_2BIT", "warning": "empty DNA"}
        header = dna[0]
        pad = 1 if header == "C" else 0
        body = dna[1:]
        bits = "".join(BASE2BITS.get(ch, "00") for ch in body)
        if pad:
            bits = bits[:-pad] if len(bits) >= pad else ""
        data, pad_to_byte = bitstring_to_bytes(bits, pad_to_byte=True)
        meta = {
            "mapping": mapping,
            "display": display_mapping(mapping),
            "mode": "SM_2BIT",
            "decoded_dna_len": len(dna),
            "bits_len": len(bits),
            "bytes_len": len(data),
            "header_pad_bits": pad,
            "pad_bits_to_byte": pad_to_byte,
        }
        return data, bits, meta

    if mapping == "RINF_B16":
        if len(dna) % 2:
            dna = dna[:-1]
        nibbles: List[int] = []
        invalid_dimers = 0
        for i in range(0, len(dna), 2):
            dimer = dna[i:i + 2]
            if dimer in DIMER2VAL:
                nibbles.append(DIMER2VAL[dimer])
            else:
                invalid_dimers += 1
                nibbles.append(0)
        if len(nibbles) % 2:
            nibbles = nibbles[:-1]
        out = bytearray()
        for i in range(0, len(nibbles), 2):
            out.append(((nibbles[i] & 0xF) << 4) | (nibbles[i + 1] & 0xF))
        data = bytes(out)
        bits = bytes_to_bitstring(data)
        meta = {
            "mapping": mapping,
            "display": display_mapping(mapping),
            "mode": "RINF_DIRECT_DIMER",
            "decoded_dna_len": len(dna),
            "bits_len": len(bits),
            "bytes_len": len(data),
            "invalid_dimers": invalid_dimers,
        }
        return data, bits, meta

    raise ValueError(f"Unsupported mapping: {mapping}")
