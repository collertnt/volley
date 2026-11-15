"""Contact sheet of every player track, for filling in the corrections file."""
from __future__ import annotations

import cv2
import numpy as np

from .video import FrameSource


def review_sheet(src, tracks, proc_width=960, tile=(110, 190), cols=8):
    """One tile per track: its best (largest) view, labelled with the track id,
    side and the number the reader settled on ('?' if none)."""
    source = src if isinstance(src, FrameSource) else FrameSource(src, proc_width=proc_width)
    want = {}
    for ident in tracks.identities.values():
        if not ident.bbox:
            continue
        f, bb = max(ident.bbox.items(), key=lambda kv: kv[1][2] * kv[1][3])
        want.setdefault(f, []).append((ident, bb))
    if not want:
        return None
    tiles = {}
    last = max(want)
    for i, frame in enumerate(source):
        if i > last:
            break
        for ident, (x, y, w, h) in want.get(i, []):
            pad = int(0.25 * w) + 2
            crop = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
            if crop.size == 0:
                continue
            s = min(tile[0] / crop.shape[1], (tile[1] - 34) / crop.shape[0])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            t = np.full((tile[1], tile[0], 3), 40, np.uint8)
            x0 = (tile[0] - crop.shape[1]) // 2
            t[34:34 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
            num = "?" if ident.number is None else str(ident.number)
            cv2.putText(t, f"T{ident.id} {ident.team or '?'}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(t, f"number: {num}", (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (120, 230, 120) if ident.number is not None else (120, 170, 255), 1, cv2.LINE_AA)
            tiles[ident.id] = t
    order = [tiles[k] for k in sorted(tiles)]
    if not order:
        return None
    rows = []
    for r in range(0, len(order), cols):
        row = order[r:r + cols]
        row += [np.full_like(order[0], 40)] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)
