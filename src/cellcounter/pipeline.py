"""バッチ処理本体。GUI / CLI のどちらからも同じ関数を呼ぶ。

emit(kind, **data) でイベントを通知する:
  log(message) / start(total, files) / result(row) / progress(done, total) / finished(summary)
"""
from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .analysis import count_cells, make_overlay, measure_cells, relabel_sequential, split_edge_cells, to_gray
from .config import Settings
from .imagefile import list_images, read_image, write_label_image, write_png

Emit = Callable[..., None]

SUMMARY_HEADER = [
    "FileName", "CellCount", "FilteredCellCount", "EdgeRemovedCount",
    "MeanArea_px", "MedianArea_px", "Seconds", "Error",
]
CELL_HEADER = [
    "FileName", "Label", "Area_px", "Area_um2", "CentroidY", "CentroidX",
    "MeanIntensity", "EdgeRemoved",
]


class _CsvSink:
    """1 行ごとに flush する CSV。途中で止めても結果が残る。
    Excel でロックされていたら日時付きの別名で保存する（旧版と同じ挙動）。"""

    def __init__(self, path: Path, header: list[str], log: Callable[[str], None]):
        try:
            self.f = open(path, "w", newline="", encoding="utf-8-sig")
        except PermissionError:
            alt = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
            log(f"{path.name} は他のアプリで開かれているため {alt.name} に保存します")
            path = alt
            self.f = open(path, "w", newline="", encoding="utf-8-sig")
        self.path = path
        self.w = csv.writer(self.f)
        self.w.writerow(header)
        self.f.flush()

    def write(self, rows):
        self.w.writerows(rows)
        self.f.flush()

    def close(self):
        self.f.close()


def seg_path(out_dir: Path, src: Path) -> Path:
    """{元ファイル名から拡張子だけ除いたもの}_seg.npy（例: 0.1_test.tif → 0.1_test_seg.npy）"""
    return out_dir / f"{src.stem}_seg.npy"


def _save_seg(out_dir: Path, src: Path, image, masks, flows, log) -> Path:
    """Cellpose GUI で開ける _seg.npy を保存。

    Cellpose の masks_flows_to_seg は渡されたパスに os.path.splitext をかけて
    "_seg.npy" を付ける。拡張子なしのパス（…/0.1_test）を渡すと ".1_test" が
    拡張子とみなされ 0_seg.npy になってしまう（旧版のバグ）。
    そのため拡張子付きのファイル名（…/0.1_test.tif）を渡し、保存先も検証する。
    """
    target = seg_path(out_dir, src)
    try:
        from cellpose import io as cpio

        cpio.masks_flows_to_seg(image, masks, flows, str(out_dir / src.name))
        if not target.exists():
            raise FileNotFoundError(f"{target.name} が作成されませんでした")
        # Cellpose GUI が元画像を見つけられるよう、filename を入力画像のパスにする
        dat = np.load(str(target), allow_pickle=True).item()
        dat["filename"] = str(src)
        np.save(str(target), dat, allow_pickle=True)
    except Exception as e:  # バージョン差異などで失敗したら最低限の内容で保存
        log(f"  seg.npy を簡易形式で保存します（{e!r}）")
        np.save(str(target), {"masks": masks, "filename": str(src)}, allow_pickle=True)
    return target


def process_image(path: Path, segmenter, s: Settings, out_dir: Path, log) -> tuple[dict, list]:
    image = read_image(path)
    masks, flows = segmenter.segment(image)
    masks = np.asarray(masks)
    if masks.ndim != 2:
        raise ValueError(f"想定外のマスク形状: {masks.shape}")
    masks = masks.astype(np.int32, copy=False)

    if s.edge_filter:
        kept, removed, removed_ids = split_edge_cells(masks, s.edges, s.edge_margin)
    else:
        kept, removed, removed_ids = masks, np.zeros_like(masks), np.zeros(0, dtype=masks.dtype)

    table = measure_cells(masks, to_gray(image), removed_ids)
    kept_area = table.area_px[~table.edge_removed]

    base = out_dir / path.stem
    if s.save_overlay:
        write_png(base.with_name(base.name + "_overlay.png"), make_overlay(image, kept, removed, s.overlay_background))
    if s.save_masks:
        write_label_image(base, masks)
    if s.save_seg:
        _save_seg(out_dir, path, image, masks, flows, log)

    row = {
        "FileName": path.name,
        "CellCount": count_cells(masks),
        "FilteredCellCount": count_cells(relabel_sequential(kept)),
        "EdgeRemovedCount": int(removed_ids.size),
        "MeanArea_px": round(float(kept_area.mean()), 2) if kept_area.size else "",
        "MedianArea_px": round(float(np.median(kept_area)), 2) if kept_area.size else "",
    }
    px2 = s.pixel_size_um ** 2 if s.pixel_size_um > 0 else None
    cell_rows = [
        [path.name, lbl, area, round(area * px2, 4) if px2 else "", round(cy, 2), round(cx, 2),
         round(mi, 3), int(edge)]
        for lbl, area, cy, cx, mi, edge in table.rows()
    ]
    return row, cell_rows


def run_batch(
    s: Settings,
    emit: Emit,
    should_stop: Callable[[], bool] = lambda: False,
    segmenter_factory: Callable | None = None,
    env_info: dict | None = None,
) -> dict:
    log = lambda m: emit("log", message=m)  # noqa: E731

    in_dir = Path(s.input_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"画像フォルダが見つかりません: {in_dir}")
    out_dir = s.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(in_dir)
    emit("start", total=len(files), files=[f.name for f in files])
    if not files:
        log("対象画像がありません（tif / png / jpg / bmp）")

    started = datetime.now()
    record = {
        "app": "CellCounter", "app_version": __version__,
        "started": started.isoformat(timespec="seconds"),
        "finished": None, "status": "running",
        "input_dir": str(in_dir.resolve()), "output_dir": str(out_dir.resolve()),
        "settings": s.to_dict(), "environment": env_info or {},
    }
    record_path = out_dir / "settings.json"

    def write_record():
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    write_record()

    if segmenter_factory is None:
        from .segmenter import CellposeSegmenter, LogForwarder

        logging.getLogger("cellpose").addHandler(LogForwarder(log))
        segmenter_factory = lambda st: CellposeSegmenter(st, log)  # noqa: E731

    t0 = time.perf_counter()
    segmenter = segmenter_factory(s)
    record["model"] = getattr(segmenter, "model_name", "")
    record["device"] = getattr(segmenter, "device", "")
    log(f"モデル準備完了（{time.perf_counter() - t0:.1f} 秒） / 実行デバイス: {record['device']}")
    write_record()

    summary_csv = _CsvSink(out_dir / "image_summary.csv", SUMMARY_HEADER, log)
    cell_csv = _CsvSink(out_dir / "cell_measurements.csv", CELL_HEADER, log) if s.save_cell_table else None

    n_done = n_err = total_cells = total_filtered = 0
    status = "completed"
    try:
        for i, path in enumerate(files):
            if should_stop():
                status = "cancelled"
                log("停止しました")
                break
            log(f"[{i + 1}/{len(files)}] {path.name}")
            t = time.perf_counter()
            try:
                row, cell_rows = process_image(path, segmenter, s, out_dir, log)
                row["Error"] = ""
                if cell_csv:
                    cell_csv.write(cell_rows)
                total_cells += row["CellCount"]
                total_filtered += row["FilteredCellCount"]
            except Exception as e:
                n_err += 1
                row = {"FileName": path.name, "CellCount": "", "FilteredCellCount": "",
                       "EdgeRemovedCount": "", "MeanArea_px": "", "MedianArea_px": "", "Error": str(e)}
                log(f"  エラー: {e}")
            row["Seconds"] = round(time.perf_counter() - t, 2)
            summary_csv.write([[row[h] for h in SUMMARY_HEADER]])
            n_done += 1
            if s.save_overlay and not row["Error"]:
                row["overlay"] = str(out_dir / f"{path.stem}_overlay.png")
            emit("result", row=row)
            emit("progress", done=n_done, total=len(files))
    except BaseException:
        status = "failed"
        raise
    finally:
        summary_csv.close()
        if cell_csv:
            cell_csv.close()
        record.update(
            finished=datetime.now().isoformat(timespec="seconds"), status=status,
            n_images=len(files), n_processed=n_done, n_errors=n_err,
            total_cells=total_cells, total_filtered_cells=total_filtered,
            summary_csv=summary_csv.path.name,
        )
        write_record()

    summary = {
        "status": status, "processed": n_done, "total": len(files), "errors": n_err,
        "total_cells": total_cells, "total_filtered_cells": total_filtered,
        "seconds": round(time.perf_counter() - t0, 1), "output_dir": str(out_dir),
        "summary_csv": str(summary_csv.path),
    }
    emit("finished", summary=summary)
    return summary
