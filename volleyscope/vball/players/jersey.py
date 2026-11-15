"""Jersey number reading.

Classical, CPU-only reader: segment the digits against the jersey colour,
classify each digit by nearest neighbour against digits rendered from the
fonts available offline (OpenCV's built-in fonts + the DejaVu fonts that ship
with matplotlib). An optional Tesseract backend can be used if installed.

This is the weakest link on real footage: real jerseys use block fonts,
wrinkle, and are small in frame. It returns None rather than guess, and the
corrections file exists for the ones it gets wrong.
"""
from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache

import cv2
import numpy as np

MIN_DIGIT_PX = 12         # below this digit height nothing is attempted
_W, _H = 24, 32


def _normalise(glyph: np.ndarray) -> np.ndarray:
    """Binary glyph -> fixed size, aspect preserved (so a '1' stays thin)."""
    ys, xs = np.where(glyph > 0)
    g = glyph[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    h, w = g.shape
    s = min((_W - 2) / w, (_H - 2) / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((_H, _W), np.uint8)
    x0, y0 = (_W - nw) // 2, (_H - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = g
    return out


def _feature(img: np.ndarray) -> np.ndarray:
    f = cv2.resize(cv2.GaussianBlur(img, (3, 3), 0), (12, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    f -= f.mean()
    return f / (np.linalg.norm(f) + 1e-6)


def _ttf_fonts():
    try:
        import matplotlib
        d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        names = ["DejaVuSans-Bold.ttf", "DejaVuSansCondensed-Bold.ttf", "DejaVuSerif-Bold.ttf",
                 "DejaVuSansMono-Bold.ttf", "DejaVuSans.ttf"]
        return [os.path.join(d, n) for n in names if os.path.exists(os.path.join(d, n))]
    except Exception:
        return []


def render_digit(d: int, font, thick=3) -> np.ndarray:
    """Render one digit as a binary image. `font` is an OpenCV font id or a .ttf path."""
    if isinstance(font, str):
        from PIL import Image, ImageDraw, ImageFont
        im = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(im).text((20, 4), str(d), fill=255, font=ImageFont.truetype(font, 72))
        return (np.array(im) > 127)
    img = np.zeros((96, 96), np.uint8)
    cv2.putText(img, str(d), (20, 76), font, 2.6, 255, thick, cv2.LINE_AA)
    return img > 127


ALL_FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX,
             cv2.FONT_HERSHEY_COMPLEX] + _ttf_fonts()


@lru_cache(maxsize=4)
def _templates(exclude: tuple = ()):
    feats, labels = [], []
    rng = np.random.default_rng(0)
    for font in ALL_FONTS:
        if font in exclude:
            continue
        for d in range(10):
            variants = [render_digit(d, font, t) for t in ((3, 6, 9) if not isinstance(font, str) else (0,))]
            for g in variants:
                g8 = g.astype(np.uint8) * 255
                for k in range(5):
                    a = g8
                    if k:
                        M = cv2.getRotationMatrix2D((48, 48), rng.uniform(-7, 7), rng.uniform(0.9, 1.1))
                        M[0, 1] += rng.uniform(-0.12, 0.12)
                        a = cv2.warpAffine(g8, M, (96, 96))
                        if k == 3:
                            a = cv2.dilate(a, np.ones((3, 3), np.uint8))
                        if k == 4:
                            a = cv2.erode(a, np.ones((3, 3), np.uint8))
                    if (a > 127).sum() < 20:
                        continue
                    feats.append(_feature(_normalise(a > 127)))
                    labels.append(d)
    return np.array(feats), np.array(labels)


def classify_digit(glyph: np.ndarray, exclude: tuple = ()):
    """Returns (digit, confidence 0..1)."""
    F, Lb = _templates(exclude)
    sim = F @ _feature(_normalise(glyph))
    best = np.zeros(10)
    for d in range(10):
        best[d] = sim[Lb == d].max()
    order = np.argsort(-best)
    top, second = best[order[0]], best[order[1]]
    conf = float(np.clip((top - 0.45) / 0.4, 0, 1) * np.clip((top - second) / 0.08, 0, 1))
    return int(order[0]), conf


def torso_crop(frame, bbox):
    x, y, w, h = bbox
    y0, y1 = y + int(0.16 * h), y + int(0.54 * h)
    x0, x1 = x + int(0.08 * w), x + int(0.92 * w)
    return frame[max(0, y0):y1, max(0, x0):x1]


def segment_digits(crop: np.ndarray):
    """Find digit-shaped regions that contrast with the dominant jersey colour.

    Returns a list of binary glyphs, left to right (at most two)."""
    h, w = crop.shape[:2]
    if h < MIN_DIGIT_PX + 2 or w < 6:
        return []
    up = max(1, int(np.ceil(40 / h)))
    big = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    H, W = big.shape[:2]
    centre = big[H // 5:4 * H // 5, W // 4:3 * W // 4].reshape(-1, 3).astype(np.float32)
    # dominant jersey colour = the densest colour in the middle of the torso
    q = (centre // 32).astype(np.int32)
    keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    dom = centre[keys == Counter(keys.tolist()).most_common(1)[0][0]].mean(0)
    dist = np.linalg.norm(big.astype(np.float32) - dom, axis=2)
    d8 = np.clip(dist, 0, 255).astype(np.uint8)
    thr, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr < 35:                                   # nothing contrasts with the jersey
        return []
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cands = []
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if x <= 0 or x + cw >= W:                  # touches the side: an arm, not a digit
            continue
        if ch < 0.25 * H or ch > 0.97 * H or ch < MIN_DIGIT_PX * up:
            continue
        if not (0.1 <= cw / ch <= 1.05) or area < 0.15 * cw * ch:
            continue
        cands.append((x, y, cw, ch, i))
    if not cands:
        return []
    tall = max(c[3] for c in cands)
    ref = max(cands, key=lambda c: c[3])
    cands = [c for c in cands if c[3] >= 0.75 * tall and abs((c[1] + c[3] / 2) - (ref[1] + ref[3] / 2)) < 0.25 * tall]
    cands.sort(key=lambda c: abs(c[0] + c[2] / 2 - W / 2))
    cands = sorted(cands[:2], key=lambda c: c[0])
    if len(cands) == 2 and cands[1][0] - (cands[0][0] + cands[0][2]) > 0.8 * tall:
        cands = cands[:1]                          # too far apart to be one number
    # a number sits in the middle of the jersey; a lone off-centre digit is
    # usually half of a two-digit number whose other half was not segmented
    left, right = cands[0][0], cands[-1][0] + cands[-1][2]
    if abs((left + right) / 2 - W / 2) > 0.14 * W:
        return []
    return [(lab[y:y + ch, x:x + cw] == i) for x, y, cw, ch, i in cands]


def read_number(frame, bbox, min_conf=0.35, backend="template", exclude=()):
    """Read the jersey number of the player in `bbox`.

    Returns (number, confidence) or (None, 0.0) when not legible."""
    crop = torso_crop(frame, bbox)
    if crop.size == 0:
        return None, 0.0
    if backend == "tesseract":
        return _read_tesseract(crop)
    glyphs = segment_digits(crop)
    if not glyphs:
        return None, 0.0
    digits, confs = zip(*(classify_digit(g, exclude) for g in glyphs))
    conf = float(min(confs))
    if conf < min_conf:
        return None, conf
    return int("".join(map(str, digits))), conf


def _read_tesseract(crop):
    """Optional backend; needs the tesseract binary + pytesseract installed."""
    import pytesseract
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = 255 - bw
    txt = pytesseract.image_to_string(255 - bw, config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
    if txt.isdigit() and 1 <= len(txt) <= 2:
        return int(txt), 0.6
    return None, 0.0


class NumberVotes:
    """Accumulates per-frame reads into one number per track segment."""

    def __init__(self):
        self.votes = Counter()
        self.n = 0

    def add(self, number, conf):
        if number is not None:
            self.votes[int(number)] += conf
            self.n += 1

    def result(self, min_reads=6, min_share=0.6):
        """(number, confidence) or (None, 0.0)."""
        if self.n < min_reads or not self.votes:
            return None, 0.0
        total = sum(self.votes.values())
        (top, wt), = self.votes.most_common(1)
        # a two-digit number half hidden reads as one of its digits: prefer the
        # two-digit reading when it has real support
        if top < 10:
            for num, w2 in self.votes.items():
                if num >= 10 and str(top) in str(num) and w2 >= 0.3 * total:
                    top, wt = num, w2 + wt
                    break
        share = wt / total
        if share < min_share:
            return None, float(share)
        return int(top), float(min(1.0, share * min(1.0, self.n / 8)))
