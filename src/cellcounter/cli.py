"""コマンドライン:  cellcounter run / info / gui"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cellcounter", description="Cellpose による細胞カウント")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("gui", help="GUI を起動（引数なしでも GUI）")
    sub.add_parser("info", help="Python / Cellpose / GPU の環境チェック")

    r = sub.add_parser("run", help="フォルダ内の画像を一括解析")
    r.add_argument("input_dir")
    r.add_argument("output_dir", nargs="?", default="", help="省略時は <input_dir>/results")
    r.add_argument("--config", help="設定 JSON（前回の settings.json も可）")
    r.add_argument("--model", dest="model_path")
    r.add_argument("--cpu", action="store_true", help="GPU を使わない")
    r.add_argument("--diameter", type=float)
    r.add_argument("--flow-threshold", type=float)
    r.add_argument("--cellprob-threshold", type=float)
    r.add_argument("--min-size", type=int)
    r.add_argument("--edges", help="除去する端（例: top,left）。none で無効")
    r.add_argument("--edge-margin", type=int)
    r.add_argument("--no-overlay", action="store_true")
    r.add_argument("--overlay-background", choices=["gray", "color"],
                   help="オーバーレイの背景（gray = グレースケール [既定] / color = 元の色）")
    r.add_argument("--save-masks", action="store_true")
    r.add_argument("--save-seg", action="store_true")
    r.add_argument("--pixel-size", type=float, dest="pixel_size_um", help="µm/px")
    return p


def _settings_from_args(a) -> Settings:
    s = Settings.load(Path(a.config)) if a.config else Settings()
    s.input_dir = a.input_dir
    s.output_dir = a.output_dir or (s.output_dir if a.config else "")
    for k in ("model_path", "diameter", "flow_threshold", "cellprob_threshold", "min_size",
              "edge_margin", "pixel_size_um"):
        v = getattr(a, k)
        if v is not None:
            setattr(s, k, v)
    if a.cpu:
        s.use_gpu = False
    if a.edges is not None:
        if a.edges.lower() in ("none", ""):
            s.edge_filter = False
        else:
            s.edge_filter = True
            s.edges = [e.strip() for e in a.edges.split(",") if e.strip()]
    if a.no_overlay:
        s.save_overlay = False
    if a.overlay_background:
        s.overlay_background = a.overlay_background
    if a.save_masks:
        s.save_masks = True
    if a.save_seg:
        s.save_seg = True
    return s


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd in (None, "gui"):
        from .gui.app import main as gui_main

        return gui_main()

    from .segmenter import environment_info

    if args.cmd == "info":
        print(json.dumps(environment_info(), ensure_ascii=False, indent=2))
        return 0

    from .pipeline import run_batch

    s = _settings_from_args(args)

    def emit(kind, **d):
        if kind == "log":
            print(d["message"], flush=True)
        elif kind == "result":
            r = d["row"]
            if not r.get("Error"):
                print(f"  → {r['CellCount']} 個（境界除去後 {r['FilteredCellCount']} 個） {r['Seconds']} 秒")
        elif kind == "finished":
            sm = d["summary"]
            print(f"\n完了: {sm['processed']}/{sm['total']} 枚, エラー {sm['errors']} 件, {sm['seconds']} 秒")
            print(f"結果: {sm['summary_csv']}")

    try:
        summary = run_batch(s, emit, env_info=environment_info())
    except KeyboardInterrupt:
        print("\n中断しました（ここまでの結果は CSV に保存済み）")
        return 130
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    return 0 if summary["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
