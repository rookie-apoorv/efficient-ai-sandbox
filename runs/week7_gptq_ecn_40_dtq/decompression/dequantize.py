"""Inverse of codec E (see compression/codec.py). numpy + torch only, CPU."""

from __future__ import annotations

import numpy as np
import torch

from decompression.rans import decode

ROW_CHUNK = 8192


def restore_tensor(store, name: str, meta: dict) -> torch.Tensor:
    rows, cols = int(meta["rows"]), int(meta["cols"])
    G = int(meta["group_size"])
    n = rows * cols
    # streams are stored as signed int views of unsigned data; reinterpret
    dts = meta["stream_dtypes"]
    sym = decode(
        store.get(f"{name}::words").numpy().view(np.dtype(dts["words"])),
        store.get(f"{name}::state").numpy().view(np.dtype(dts["state"])),
        store.get(f"{name}::count").numpy().view(np.dtype(dts["count"])),
        store.get(f"{name}::freq").numpy().view(np.dtype(dts["freq"])),
        n,
    )
    esc = int(meta["escape"])
    qmin = int(meta["q_min"])
    is_esc = sym == esc
    sym2d = torch.from_numpy(sym).reshape(rows, cols)
    steps = store.get(f"{name}::steps").float()          # [rows, cols // G]
    out = torch.empty((rows, cols), dtype=torch.float32)
    for r0 in range(0, rows, ROW_CHUNK):
        r1 = min(r0 + ROW_CHUNK, rows)
        out[r0:r1] = (sym2d[r0:r1].float() + qmin) * steps[r0:r1].repeat_interleave(G, 1)
    n_out = int(meta.get("outliers", 0))
    if n_out:
        vals = store.get(f"{name}::outliers").float()
        if vals.numel() != n_out or int(is_esc.sum()) != n_out:
            raise ValueError(f"{name}: outlier count mismatch")
        out.view(-1)[torch.from_numpy(np.flatnonzero(is_esc))] = vals
    dtype = getattr(torch, meta["dtype"])
    return out.reshape(tuple(meta["shape"])).to(dtype).contiguous()
