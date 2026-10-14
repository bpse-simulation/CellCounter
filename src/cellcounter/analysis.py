"""マスク後処理（境界細胞除去・計測・輪郭オーバーレイ）。

Cellpose に依存しない純粋な numpy 処理なので、単体テストが可能。
旧版は細胞ラベルごとに全画素をループしていた（O(細胞数 × 画素数)）が、
ここではすべてベクトル化している。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EDGES = ("top", "bottom", "left", "right")


# ---------------------------------------------------------------------------
# 境界細胞除去
# ---------------------------------------------------------------------------
def edge_labels(masks: np.ndarray, edges: tuple[str, ...] | list[str], margin: int) -> np.ndarray:
    """指定した画像端から margin px 以内に 1 画素でもかかる細胞ラベルを返す。"""
    if not edges:
        return np.zeros(0, dtype=masks.dtype)
    h, w = masks.shape
    m = max(1, int(margin))  # 0 以下は「端に接している」= 1px として扱う
    strips = []
    if "top" in edges:
        strips.append(masks[: min(m, h), :].ravel())
    if "bottom" in edges:
        strips.append(masks[max(h - m, 0):, :].ravel())
    if "left" in edges:
        strips.append(masks[:, : min(m, w)].ravel())
    if "right" in edges:
        strips.append(masks[:, max(w - m, 0):].ravel())
    labels = np.unique(np.concatenate(strips))
    return labels[labels != 0]


def split_edge_cells(masks: np.ndarray, edges, margin: int):
    """masks を (採用マスク, 除去マスク, 除去ラベル) に分ける。ラベル番号は元のまま。"""
    removed_ids = edge_labels(masks, edges, margin)
    if removed_ids.size == 0:
        return masks, np.zeros_like(masks), removed_ids
    is_removed = np.isin(masks, removed_ids)
    kept = np.where(is_removed, 0, masks)
    removed = np.where(is_removed, masks, 0)
    return kept, removed, removed_ids


def relabel_sequential(masks: np.ndarray) -> np.ndarray:
    """ラベルを 1..N に振り直す。"""
    ids = np.unique(masks)
    ids = ids[ids != 0]
    if ids.size == 0:
        return np.zeros_like(masks)
    lut = np.zeros(int(masks.max()) + 1, dtype=np.int32 if ids.size < 2**31 else np.int64)
    lut[ids] = np.arange(1, ids.size + 1)
    return lut[masks]


def count_cells(masks: np.ndarray) -> int:
    ids = np.unique(masks)
    return int(np.count_nonzero(ids))


# ---------------------------------------------------------------------------
# 細胞ごとの計測
# ---------------------------------------------------------------------------
@dataclass
class CellTable:
    label: np.ndarray
    area_px: np.ndarray
    centroid_y: np.ndarray
    centroid_x: np.ndarray
    mean_intensity: np.ndarray
    edge_removed: np.ndarray  # bool

    def __len__(self) -> int:
        return int(self.label.size)

    def rows(self):
        for i in range(len(self)):
            yield (
                int(self.label[i]),
                int(self.area_px[i]),
                float(self.centroid_y[i]),
                float(self.centroid_x[i]),
                float(self.mean_intensity[i]),
                bool(self.edge_removed[i]),
            )


def measure_cells(masks: np.ndarray, intensity: np.ndarray, removed_ids: np.ndarray) -> CellTable:
    """bincount による一括計測（面積・重心・平均輝度）。"""
    flat = masks.ravel().astype(np.int64)
    n = int(flat.max()) + 1 if flat.size else 1
    area = np.bincount(flat, minlength=n)
    ids = np.nonzero(area)[0]
    ids = ids[ids != 0]

    h, w = masks.shape
    yy, xx = np.indices((h, w))
    sum_y = np.bincount(flat, weights=yy.ravel(), minlength=n)
    sum_x = np.bincount(flat, weights=xx.ravel(), minlength=n)
    sum_i = np.bincount(flat, weights=intensity.astype(np.float64).ravel(), minlength=n)

    a = area[ids].astype(np.float64)
    return CellTable(
        label=ids,
        area_px=area[ids],
        centroid_y=sum_y[ids] / a,
        centroid_x=sum_x[ids] / a,
        mean_intensity=sum_i[ids] / a,
        edge_removed=np.isin(ids, removed_ids),
    )


# ---------------------------------------------------------------------------
# 表示用画像・輪郭オーバーレイ
# ---------------------------------------------------------------------------
def to_gray(image: np.ndarray) -> np.ndarray:
    """計測用の輝度画像（元のビット深度の値をそのまま平均）。"""
    if image.ndim == 2:
        return image.astype(np.float32)
    return image.astype(np.float32).mean(axis=-1)


OVERLAY_BACKGROUNDS = ("gray", "color")


def _to_uint8(img: np.ndarray, low: float = 0.1, high: float = 99.9) -> np.ndarray:
    """8bit 画像はそのまま。16bit / float 画像は全チャンネル共通の 1 つの範囲で 8bit 化する。

    （チャンネルごとに別々に伸ばすと色のバランスが崩れるため、必ず共通の範囲を使う）
    """
    if img.dtype == np.uint8:
        return img
    f = img.astype(np.float32)
    lo, hi = np.percentile(f, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((f - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_display_rgb(image: np.ndarray, background: str = "gray") -> np.ndarray:
    """オーバーレイの背景にする 8bit RGB を作る。

    background="gray"  : グレースケール（既定）。位相差・明視野で輪郭の色が見やすい
    background="color" : 元の色のまま（蛍光の多チャンネル画像など、色に意味がある場合）
    """
    if image.ndim == 2:
        g = _to_uint8(image)
        return np.repeat(g[..., None], 3, axis=-1)

    if background == "gray":
        f = image.astype(np.float32)
        if image.shape[-1] == 3:  # 輝度 (ITU-R BT.601)
            lum = f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114
        else:
            lum = f.mean(axis=-1)
        lum = lum.round().astype(image.dtype) if image.dtype == np.uint8 else lum
        g = _to_uint8(lum)
        return np.repeat(g[..., None], 3, axis=-1)

    rgb = _to_uint8(image)
    if rgb.shape[-1] == 2:  # 2ch は R/G に割り当て
        rgb = np.concatenate([rgb, np.zeros_like(rgb[..., :1])], axis=-1)
    return np.ascontiguousarray(rgb[..., :3])


def outlines(masks: np.ndarray) -> np.ndarray:
    """各細胞の内側 1px 輪郭（4 近傍でラベルが変わる画素）。"""
    b = np.zeros(masks.shape, dtype=bool)
    diff_v = masks[:-1, :] != masks[1:, :]
    diff_h = masks[:, :-1] != masks[:, 1:]
    b[:-1, :] |= diff_v
    b[1:, :] |= diff_v
    b[:, :-1] |= diff_h
    b[:, 1:] |= diff_h
    b[0, :] = b[-1, :] = True
    b[:, 0] = b[:, -1] = True
    return b & (masks > 0)


KEEP_COLOR = (0, 255, 0)     # 緑 = 採用
REMOVED_COLOR = (255, 0, 0)  # 赤 = 境界除去


def make_overlay(image: np.ndarray, kept: np.ndarray, removed: np.ndarray | None = None,
                 background: str = "gray") -> np.ndarray:
    """RGB（R,G,B 順）のオーバーレイ画像を返す。"""
    rgb = to_display_rgb(image, background).copy()
    rgb[outlines(kept)] = KEEP_COLOR
    if removed is not None and removed.any():
        rgb[outlines(removed)] = REMOVED_COLOR
    return rgb
