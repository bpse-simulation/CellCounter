"""Cellpose ラッパー。モデルはバッチ全体で 1 回だけ読み込む。"""
from __future__ import annotations

import inspect
import logging
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Protocol

import numpy as np


class Segmenter(Protocol):
    device: str
    model_name: str

    def segment(self, image: np.ndarray) -> tuple[np.ndarray, object]:
        """(masks, flows) を返す。"""
        ...


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def detect_device() -> tuple[str, str]:
    """('cuda' | 'mps' | 'cpu', 表示名)"""
    try:
        import torch
    except Exception:
        return "cpu", "CPU (PyTorch なし)"
    try:
        if torch.cuda.is_available():
            return "cuda", f"CUDA: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", "Apple MPS"
    except Exception:
        pass
    return "cpu", "CPU"


def environment_info() -> dict:
    """起動時チェック・settings.json 用の環境情報。"""
    info: dict = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cellpose": _pkg_version("cellpose"),
        "torch": _pkg_version("torch"),
        "numpy": _pkg_version("numpy"),
        "cellpose_import_ok": False,
        "gpu_available": False,
        "device": "cpu",
        "device_name": "CPU",
        "error": None,
    }
    try:
        import cellpose.models  # noqa: F401
        info["cellpose_import_ok"] = True
    except Exception as e:  # pragma: no cover - 実環境依存
        info["error"] = f"cellpose を import できません: {e!r}"
        return info
    dev, name = detect_device()
    info.update(device=dev, device_name=name, gpu_available=dev != "cpu")
    return info


def build_eval_kwargs(settings, accepted: set[str]) -> dict:
    """設定から model.eval() の引数を作る。

    Cellpose のバージョンによって eval の引数が違うので、accepted（eval が受け取る引数名）
    にあるものだけ渡す。
    """
    params = {
        "diameter": settings.diameter if settings.diameter and settings.diameter > 0 else None,
        "flow_threshold": float(settings.flow_threshold),
        "cellprob_threshold": float(settings.cellprob_threshold),
        "min_size": int(settings.min_size),
        # Cellpose GUI の norm percentiles に相当（dict で渡すと Cellpose の既定値とマージされる）
        "normalize": {"percentile": [float(settings.norm_percentile_low),
                                     float(settings.norm_percentile_high)]},
        # 0 / None = 自動（直径に比例）。細長い細胞では大きく（例: 2000）すると輪郭が途切れにくい
        "niter": int(settings.niter) if settings.niter and settings.niter > 0 else None,
    }
    return {k: v for k, v in params.items() if k in accepted}


class CellposeSegmenter:
    def __init__(self, settings, log: Callable[[str], None] = print):
        from cellpose import models

        dev, dev_name = detect_device()
        gpu = bool(settings.use_gpu) and dev != "cpu"
        self.device = dev_name if gpu else "CPU"

        kwargs = {"gpu": gpu}
        if settings.model_path:
            kwargs["pretrained_model"] = settings.model_path
            self.model_name = settings.model_path
        else:
            self.model_name = f"Cellpose default (cellpose {_pkg_version('cellpose')})"

        log(f"モデル読み込み中: {self.model_name} / {self.device}")
        self.model = models.CellposeModel(**kwargs)

        accepted = set(inspect.signature(self.model.eval).parameters)
        self.eval_kwargs = build_eval_kwargs(settings, accepted)

    def segment(self, image: np.ndarray):
        out = self.model.eval(image, **self.eval_kwargs)
        masks = np.asarray(out[0])
        flows = out[1] if len(out) > 1 else None
        return np.squeeze(masks), flows


class LogForwarder(logging.Handler):
    """cellpose のロガー出力を GUI ログへ流す（WARNING 以上）。"""

    def __init__(self, log: Callable[[str], None]):
        super().__init__(level=logging.WARNING)
        self._log = log

    def emit(self, record):  # pragma: no cover
        try:
            self._log(f"[cellpose] {record.getMessage()}")
        except Exception:
            pass
