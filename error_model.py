from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from dna_design import BASES, clean_dna


def mutate_dna(
    seq: str,
    *,
    substitution_rate: float = 0.001,
    insertion_rate: float = 0.0,
    deletion_rate: float = 0.0,
    seed: int = 7,
    allow_indels: bool = False,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    seq = clean_dna(seq)
    rng = random.Random(int(seed))
    out: List[str] = []
    events: List[Dict[str, Any]] = []
    sub_count = ins_count = del_count = 0
    read_pos = 0

    for pos, base in enumerate(seq, start=1):
        if allow_indels and rng.random() < float(deletion_rate):
            del_count += 1
            events.append({
                "Original position": pos,
                "Read position": read_pos + 1,
                "Operation": "deletion",
                "Original base": base,
                "New/inserted base": "",
            })
            if rng.random() < float(insertion_rate):
                nb = rng.choice(BASES)
                out.append(nb)
                read_pos += 1
                ins_count += 1
                events.append({
                    "Original position": pos,
                    "Read position": read_pos,
                    "Operation": "insertion",
                    "Original base": "",
                    "New/inserted base": nb,
                })
            continue

        new_base = base
        if rng.random() < float(substitution_rate):
            choices = [b for b in BASES if b != base]
            new_base = rng.choice(choices)
            sub_count += 1
            events.append({
                "Original position": pos,
                "Read position": read_pos + 1,
                "Operation": "substitution",
                "Original base": base,
                "New/inserted base": new_base,
            })
        out.append(new_base)
        read_pos += 1

        if allow_indels and rng.random() < float(insertion_rate):
            nb = rng.choice(BASES)
            out.append(nb)
            read_pos += 1
            ins_count += 1
            events.append({
                "Original position": pos,
                "Read position": read_pos,
                "Operation": "insertion",
                "Original base": "",
                "New/inserted base": nb,
            })

    noisy = "".join(out)
    metrics = {
        "input_dna_len": len(seq),
        "noisy_dna_len": len(noisy),
        "substitutions": sub_count,
        "insertions": ins_count,
        "deletions": del_count,
        "total_errors": sub_count + ins_count + del_count,
        "length_preserved": len(seq) == len(noisy),
    }
    return noisy, events, metrics
