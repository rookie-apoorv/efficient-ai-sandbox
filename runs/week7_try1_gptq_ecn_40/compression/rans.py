"""Interleaved rANS encoder, pure numpy. Inverse of ``decompression/rans.py``.

Stores one entropy-coded stream per tensor. The lane count is chosen so each
lane carries ~``TARGET_STEPS`` symbols: fewer lanes means less per-lane
overhead (6 bytes of state + count), more lanes means fewer Python-level steps.
"""

from __future__ import annotations

import numpy as np

PROB_BITS = 15
PROB_SCALE = 1 << PROB_BITS
RANS_L = 1 << 16
TARGET_STEPS = 16384
MAX_LANES = 1 << 15


def lanes_for(n: int) -> int:
    return int(max(1, min(MAX_LANES, n // TARGET_STEPS)))


def quantize_freqs(counts: np.ndarray) -> np.ndarray:
    """Integer frequencies summing to PROB_SCALE, every present symbol >= 1."""
    counts = counts.astype(np.float64)
    present = counts > 0
    if present.sum() > PROB_SCALE:
        raise ValueError("alphabet larger than the probability scale")
    f = np.floor(counts / counts.sum() * PROB_SCALE).astype(np.int64)
    f[present & (f == 0)] = 1
    diff = PROB_SCALE - f.sum()
    # hand the remainder to / take it from the most frequent symbols
    order = np.argsort(-counts)
    i = 0
    while diff != 0:
        j = order[i % present.sum()]
        if diff > 0:
            f[j] += 1
            diff -= 1
        elif f[j] > 1:
            f[j] -= 1
            diff += 1
        i += 1
    return f


def encode(sym: np.ndarray, n_symbols: int):
    """Encode int symbols in [0, n_symbols). Returns dict of arrays + stats."""
    sym = np.ascontiguousarray(sym, dtype=np.int64).reshape(-1)
    n = sym.size
    counts = np.bincount(sym, minlength=n_symbols)
    freq = quantize_freqs(counts)
    cum = np.zeros(n_symbols + 1, dtype=np.int64)
    np.cumsum(freq, out=cum[1:])

    L = lanes_for(n)
    T = -(-n // L)
    pad = T * L - n
    if pad:
        sym = np.concatenate([sym, np.full(pad, int(np.argmax(counts)), dtype=np.int64)])
    S = sym.reshape(T, L)

    x = np.full(L, RANS_L, dtype=np.int64)
    emit = np.zeros((T, L), dtype=bool)
    val = np.zeros((T, L), dtype=np.uint16)
    shift = 32 - PROB_BITS  # x_max = freq << (16 + 16 - PROB_BITS)
    for t in range(T - 1, -1, -1):
        s = S[t]
        f = freq[s]
        e = x >= (f << shift)
        emit[t] = e
        val[t] = (x & 0xFFFF).astype(np.uint16)
        x = np.where(e, x >> 16, x)
        x = ((x // f) << PROB_BITS) + (x % f) + cum[s]
    # decoder reads lane j's words in increasing t
    words = val.T[emit.T]
    count = emit.sum(0)
    count = count.astype(np.uint16 if T <= 0xFFFF else np.uint32)
    stored_bytes = words.nbytes + L * 4 + count.nbytes + freq.size * 4
    return {
        "words": words.astype(np.uint16),
        "state": x.astype(np.uint32),
        "count": count,
        "freq": freq.astype(np.uint32),
        "n": n,
        "bytes": stored_bytes,
    }
