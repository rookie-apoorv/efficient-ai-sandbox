"""Interleaved rANS decoder, pure numpy (no compiled dependency).

Format (one stream per tensor), written by ``compression/rans.py``:

* the symbol sequence ``s[0..n)`` is padded to ``T * L`` and dealt round-robin
  to ``L`` lanes: lane ``j`` holds ``s[j], s[j+L], s[j+2L], ...``. Decoding step
  ``t`` therefore yields the contiguous block ``s[t*L : (t+1)*L]``.
* every lane is an independent rANS coder with a 32-bit state and 16-bit
  renormalisation; probabilities are quantised to ``2**PROB_BITS``.
* stored per tensor: ``freq`` (uint32, one per symbol), ``state`` (uint32,
  final encoder state per lane), ``count`` (uint32, words per lane) and
  ``words`` (uint16, every lane's words back to back, in decode order).

Decoding runs all lanes in lockstep with numpy vector ops, so the cost is
~``T`` Python-level steps per tensor rather than one per symbol.
"""

from __future__ import annotations

import numpy as np

PROB_BITS = 15
PROB_SCALE = 1 << PROB_BITS
RANS_L = 1 << 16  # lower bound of the normalised state interval [L, L << 16)


def decode(words: np.ndarray, state: np.ndarray, count: np.ndarray,
           freq: np.ndarray, n: int) -> np.ndarray:
    """Return the ``n`` decoded symbols (int32)."""
    freq = freq.astype(np.int64)
    L = state.shape[0]
    T = -(-n // L)
    cum = np.zeros(freq.shape[0] + 1, dtype=np.int64)
    np.cumsum(freq, out=cum[1:])
    if cum[-1] != PROB_SCALE:
        raise ValueError(f"frequency table sums to {cum[-1]}, expected {PROB_SCALE}")
    slot_sym = np.repeat(np.arange(freq.shape[0], dtype=np.int64), freq)

    x = state.astype(np.int64)
    ptr = np.zeros(L, dtype=np.int64)
    ptr[1:] = np.cumsum(count.astype(np.int64))[:-1]
    words = np.concatenate([words.astype(np.int64), np.zeros(1, dtype=np.int64)])
    out = np.empty((T, L), dtype=np.int32)
    mask = PROB_SCALE - 1
    for t in range(T):
        slot = x & mask
        s = slot_sym[slot]
        x = freq[s] * (x >> PROB_BITS) + slot - cum[s]
        need = x < RANS_L
        x = np.where(need, (x << 16) | words[ptr], x)
        ptr += need
        out[t] = s
    return out.reshape(-1)[:n]
