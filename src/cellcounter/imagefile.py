"""画像の読み書き。

旧版の「全角パス不可」は cv2.imread / cv2.imwrite が Windows で Unicode パスを
扱えないことが原因。ここではファイルを Python 側でバイト列として読み書きし、
デコード/エンコードだけを OpenCV に任せることで日本語パスに対応する。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# 出力ファイルを入力と取り違えないための接尾辞
OUTPUT_SUFFIXES = ("_overlay", "_cp_masks", "_masks", "_seg")


def is_input_image(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
        return False
    stem = path.stem.lower()
    return not any(stem.endswith(s) for s in OUTPUT_SUFFIXES)


def list_images(folder: Path) -> list[Path]:
    files = [p for p in Path(folder).iterdir() if is_input_image(p)]
    return sorted(files, key=lambda p: p.name.lower())


def read_image(path: Path) -> np.ndarray:
    """(H, W) または (H, W, C) [C=2,3] の配列を返す。カラーは RGB 順。"""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        img = tifffile.imread(str(path))
    else:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"画像を読み込めません: {path.name}")
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return normalize_layout(np.asarray(img))


def normalize_layout(img: np.ndarray) -> np.ndarray:
    img = np.squeeze(img)
    # (C, H, W) → (H, W, C)
    if img.ndim == 3 and img.shape[0] <= 4 and img.shape[-1] > 4:
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 3 and img.shape[-1] == 4:  # アルファを捨てる（旧版は 1:4 を取るバグがあった）
        img = img[..., :3]
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] in (2, 3)):
        return img
    raise ValueError(f"未対応の画像形状です: {img.shape}（2D 画像のみ対応。Z スタック/タイムラプスは非対応）")


def write_png(path: Path, image: np.ndarray) -> None:
    """RGB / グレースケール / 16bit を PNG で保存（Unicode パス可）。"""
    arr = image
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise IOError(f"PNG エンコードに失敗しました: {path}")
    buf.tofile(str(path))


def write_label_image(base: Path, masks: np.ndarray) -> Path:
    """ラベル画像を保存。65535 個以下なら 16bit PNG、超えたら TIFF。"""
    if masks.max() <= np.iinfo(np.uint16).max:
        path = base.with_name(base.name + "_cp_masks.png")
        write_png(path, masks.astype(np.uint16))
    else:
        path = base.with_name(base.name + "_cp_masks.tif")
        tifffile.imwrite(str(path), masks.astype(np.uint32), compression="zlib")
    return path
