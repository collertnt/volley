"""Frame sources: a video file or an in-memory list of frames."""
from __future__ import annotations

import cv2
import numpy as np


class FrameSource:
    """Iterate frames, resized so the width is at most `proc_width`.

    All analysis runs at this resolution; pixel values in outputs refer to it.
    `scale` converts original-video pixels to processing pixels."""

    def __init__(self, src, fps: float | None = None, proc_width: int = 960):
        self.src = src
        self.is_file = isinstance(src, (str, bytes)) or hasattr(src, "__fspath__")
        if self.is_file:
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                raise IOError(f"cannot open video: {src}")
            self.fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            self.fps = float(fps or 30.0)
            h, w = src[0].shape[:2]
            self.n_frames = len(src)
        self.scale = min(1.0, proc_width / float(w))
        self.size = (int(round(w * self.scale)), int(round(h * self.scale)))

    def _fit(self, f):
        if self.scale != 1.0:
            f = cv2.resize(f, self.size, interpolation=cv2.INTER_AREA)
        return f

    def __iter__(self):
        if self.is_file:
            cap = cv2.VideoCapture(str(self.src))
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield self._fit(f)
            cap.release()
        else:
            for f in self.src:
                yield self._fit(np.asarray(f))

    def __len__(self):
        return self.n_frames
