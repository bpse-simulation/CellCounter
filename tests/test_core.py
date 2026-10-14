"""Cellpose なしで動くテスト（スタブ分割器を使用）。

    python -m pytest tests          # pytest がある場合
    python tests/test_core.py       # pytest がなくても実行可
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cellcounter.analysis import (  # noqa: E402
    count_cells, edge_labels, make_overlay, measure_cells, outlines, relabel_sequential, split_edge_cells,
)
from cellcounter.config import Settings  # noqa: E402
from cellcounter.imagefile import list_images, read_image, write_png  # noqa: E402
from cellcounter.pipeline import run_batch  # noqa: E402


def synthetic_image(h=200, w=240, centers=((10, 50), (100, 120), (150, 60), (100, 235))):
    yy, xx = np.indices((h, w))
    img = np.full((h, w), 100, np.uint16)
    for cy, cx in centers:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 12 ** 2] = 3000
    return img


class StubSegmenter:
    device = "stub"
    model_name = "threshold+label"

    def segment(self, image):
        g = image if image.ndim == 2 else image.mean(-1)
        masks, _ = ndi.label(g > g.mean() * 1.5)
        return masks.astype(np.int32), None


# ---------------------------------------------------------------- analysis
def test_edge_labels_directions():
    m = np.zeros((10, 10), int)
    m[0:2, 4:6] = 1   # top
    m[8:10, 4:6] = 2  # bottom
    m[4:6, 0:2] = 3   # left
    m[4:6, 8:10] = 4  # right
    m[4:6, 4:6] = 5   # center
    assert set(edge_labels(m, ["top"], 2)) == {1}
    assert set(edge_labels(m, ["top", "left"], 2)) == {1, 3}
    assert set(edge_labels(m, ["top", "bottom", "left", "right"], 2)) == {1, 2, 3, 4}
    assert edge_labels(m, [], 2).size == 0


def test_margin():
    m = np.zeros((20, 20), int)
    m[3:6, 8:12] = 1  # 端から 3px 離れている
    assert edge_labels(m, ["top"], 2).size == 0
    assert set(edge_labels(m, ["top"], 4)) == {1}


def test_split_and_relabel():
    m = np.zeros((10, 10), int)
    m[0:2, 0:2] = 3
    m[5:7, 5:7] = 7
    kept, removed, ids = split_edge_cells(m, ["top"], 1)
    assert list(ids) == [3]
    assert count_cells(kept) == 1 and count_cells(removed) == 1
    r = relabel_sequential(kept)
    assert r.max() == 1 and (r > 0).sum() == 4


def test_measure():
    m = np.zeros((10, 10), int)
    m[2:4, 2:6] = 1  # 8px, 重心 (2.5, 3.5)
    inten = np.zeros((10, 10))
    inten[2:4, 2:6] = 10
    t = measure_cells(m, inten, np.array([1]))
    assert list(t.area_px) == [8]
    assert abs(t.centroid_y[0] - 2.5) < 1e-9 and abs(t.centroid_x[0] - 3.5) < 1e-9
    assert t.mean_intensity[0] == 10 and t.edge_removed[0]


def test_outlines_and_overlay_colors():
    m = np.zeros((9, 9), int)
    m[2:7, 2:7] = 1
    o = outlines(m)
    assert o[2, 2] and o[2, 4] and not o[4, 4] and not o[0, 0]
    rm = np.zeros_like(m)
    rm[0:2, 0:2] = 2
    ov = make_overlay(np.random.rand(9, 9), m, rm)
    assert tuple(ov[2, 4]) == (0, 255, 0)   # 採用 = 緑
    assert tuple(ov[0, 0]) == (255, 0, 0)   # 除去 = 赤（旧版の BGR 取り違え対策）


# ---------------------------------------------------------------- I/O
def test_unicode_paths_and_rgba():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / "日本語フォルダ"
        folder.mkdir()
        rgba = np.zeros((20, 30, 4), np.uint8)
        rgba[..., 0] = 200  # R
        rgba[..., 3] = 255
        p = folder / "画像1.png"
        import cv2
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        buf.tofile(str(p))
        img = read_image(p)
        assert img.shape == (20, 30, 3) and img[0, 0, 0] == 200 and img[0, 0, 2] == 0
        write_png(folder / "x_overlay.png", img)
        assert [f.name for f in list_images(folder)] == ["画像1.png"]  # 出力ファイルは除外


# ---------------------------------------------------------------- pipeline
def test_pipeline_end_to_end():
    import tifffile

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "入力 画像"
        src.mkdir()
        tifffile.imwrite(str(src / "a.tif"), synthetic_image())
        tifffile.imwrite(str(src / "b.tif"), synthetic_image(centers=((100, 100),)))
        (src / "broken.png").write_bytes(b"not an image")

        s = Settings(input_dir=str(src), edges=["top", "right"], edge_margin=2,
                     save_masks=True, pixel_size_um=0.5)
        events = []
        summary = run_batch(s, lambda k, **kw: events.append((k, kw)),
                            segmenter_factory=lambda st: StubSegmenter())

        out = src / "results"
        assert summary["processed"] == 3 and summary["errors"] == 1
        rows = list(csv.DictReader(open(out / "image_summary.csv", encoding="utf-8-sig")))
        by = {r["FileName"]: r for r in rows}
        # a.tif: 4 個中、上端 (10,50) と右端 (100,235) が除去 → 2 個
        assert by["a.tif"]["CellCount"] == "4" and by["a.tif"]["FilteredCellCount"] == "2"
        assert by["a.tif"]["EdgeRemovedCount"] == "2"
        assert by["b.tif"]["FilteredCellCount"] == "1"
        assert by["broken.png"]["Error"]
        cells = list(csv.DictReader(open(out / "cell_measurements.csv", encoding="utf-8-sig")))
        assert len(cells) == 5 and float(cells[0]["Area_um2"]) > 0
        assert (out / "a_overlay.png").exists() and (out / "a_cp_masks.png").exists()
        rec = json.loads((out / "settings.json").read_text(encoding="utf-8"))
        assert rec["status"] == "completed" and rec["settings"]["edges"] == ["top", "right"]
        assert Settings.load(out / "settings.json").edges == ["top", "right"]  # 再実行用に読める
        kinds = [k for k, _ in events]
        assert kinds[0] == "start" and kinds[-1] == "finished"


def test_pipeline_stop():
    import tifffile

    with tempfile.TemporaryDirectory() as d:
        src = Path(d)
        for i in range(5):
            tifffile.imwrite(str(src / f"{i}.tif"), synthetic_image())
        n = {"r": 0}

        def emit(k, **kw):
            if k == "result":
                n["r"] += 1

        summary = run_batch(Settings(input_dir=str(src)), emit, should_stop=lambda: n["r"] >= 2,
                            segmenter_factory=lambda st: StubSegmenter())
        assert summary["status"] == "cancelled" and summary["processed"] == 2
        rows = list(csv.DictReader(open(src / "results" / "image_summary.csv", encoding="utf-8-sig")))
        assert len(rows) == 2


def _run_seg(fake_cellpose: bool):
    """ピリオドを含むファイル名で _seg.npy の名前が正しいか確認する。"""
    import os
    import types

    import tifffile

    saved = {k: sys.modules.get(k) for k in ("cellpose", "cellpose.io")}
    if fake_cellpose:
        # Cellpose 本体と同じく os.path.splitext で拡張子を落として _seg.npy を付ける
        cp = types.ModuleType("cellpose")
        cpio = types.ModuleType("cellpose.io")

        def masks_flows_to_seg(images, masks, flows, file_names, **kw):
            base = os.path.splitext(file_names)[0]
            np.save(base + "_seg.npy", {"masks": masks, "filename": file_names}, allow_pickle=True)

        cpio.masks_flows_to_seg = masks_flows_to_seg
        cp.io = cpio
        sys.modules["cellpose"], sys.modules["cellpose.io"] = cp, cpio
    else:
        sys.modules["cellpose"] = None  # import 失敗 → 簡易形式で保存する経路
    try:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)
            names = ["0.1_test.tif", "a.b.c.tif", "plain.tif"]
            for n in names:
                tifffile.imwrite(str(src / n), synthetic_image())
            run_batch(Settings(input_dir=str(src), save_seg=True), lambda k, **kw: None,
                      segmenter_factory=lambda st: StubSegmenter())
            out = src / "results"
            got = sorted(p.name for p in out.glob("*_seg.npy"))
            assert got == ["0.1_test_seg.npy", "a.b.c_seg.npy", "plain_seg.npy"], got
            dat = np.load(str(out / "0.1_test_seg.npy"), allow_pickle=True).item()
            assert dat["filename"] == str(src / "0.1_test.tif")
            assert dat["masks"].max() == 4
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_seg_npy_name_with_period_cellpose():
    _run_seg(fake_cellpose=True)


def test_seg_npy_name_with_period_fallback():
    _run_seg(fake_cellpose=False)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} passed")
