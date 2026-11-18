"""解析設定。GUI・CLI・settings.json（再現性ログ）で共通に使う。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Settings:
    input_dir: str = ""
    output_dir: str = ""  # 空なら <input_dir>/results

    # --- モデル / 推論 ---
    model_path: str = ""          # 空なら Cellpose のデフォルトモデル
    use_gpu: bool = True          # 使えない環境では自動で CPU
    diameter: float = 0.0         # 0 = 自動
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15            # これより小さいマスクは Cellpose が除去
    norm_percentile_low: float = 1.0    # 正規化の下限パーセンタイル（Cellpose GUI の norm percentiles）
    norm_percentile_high: float = 99.0  # 正規化の上限パーセンタイル
    niter: int = 0                # 輪郭を作る計算の反復回数。0 = 自動（直径に比例）

    # --- 境界細胞除去 ---
    edge_filter: bool = True
    edges: list[str] = field(default_factory=lambda: ["top", "left"])
    edge_margin: int = 2

    # --- 出力 ---
    save_overlay: bool = True
    overlay_background: str = "gray"  # "gray" = グレースケール / "color" = 元の色
    save_masks: bool = False
    save_seg: bool = False
    save_cell_table: bool = True  # 細胞ごとの計測 CSV
    pixel_size_um: float = 0.0    # >0 なら面積を µm² でも出力

    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        return Path(self.input_dir) / "results"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        names = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in d.items() if k in names})
        s.edges = [e for e in s.edges if e in ("top", "bottom", "left", "right")]
        if s.overlay_background not in ("gray", "color"):
            s.overlay_background = "gray"
        d = cls()
        if not (0 <= s.norm_percentile_low < s.norm_percentile_high <= 100):
            s.norm_percentile_low, s.norm_percentile_high = d.norm_percentile_low, d.norm_percentile_high
        if s.niter < 0:
            s.niter = d.niter
        return s

    @classmethod
    def load(cls, path: Path) -> "Settings":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # settings.json（実行記録）を渡された場合は "settings" キーの中身を使う
        if "settings" in data and isinstance(data["settings"], dict):
            data = data["settings"]
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def user_config_path() -> Path | None:
    """GUI の前回設定の保存先。

    アプリのフォルダ（pyproject.toml がある場所）直下の gui_settings.json に保存する。
    AppData などには書かないので、フォルダを削除すれば何も残らない。
    ソースのフォルダ以外から実行された場合（pip で通常インストールした場合など）は
    保存しない（None を返す）。
    """
    root = Path(__file__).resolve().parents[2]  # src/cellcounter/config.py → アプリのフォルダ
    if (root / "pyproject.toml").is_file():
        return root / "gui_settings.json"
    return None
