"""GUI から別プロセスで実行される処理。

推論を別プロセスに分けることで、
  * GUI が固まらない
  * 「強制停止」で推論中でも即座に止められる（プロセスごと終了）
  * モデルはバッチ開始時に 1 回だけ読み込む（旧版は画像ごとに Python を起動・モデル再読込）
"""
from __future__ import annotations

import os
import sys
import traceback


def _ensure_std_streams():
    # pythonw.exe では stdout/stderr が None になり、tqdm 等が落ちるため
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


def env_check_entry(queue):
    _ensure_std_streams()
    try:
        from ..segmenter import environment_info

        queue.put(("env", {"info": environment_info()}))
    except Exception:
        queue.put(("env", {"info": {"error": traceback.format_exc(), "cellpose_import_ok": False}}))


def batch_entry(settings_dict, env_info, queue, stop_event):
    _ensure_std_streams()

    def emit(kind, **data):
        queue.put((kind, data))

    try:
        from ..config import Settings
        from ..pipeline import run_batch

        run_batch(Settings.from_dict(settings_dict), emit, stop_event.is_set, env_info=env_info)
    except Exception as e:
        emit("fatal", message=f"{e}\n{traceback.format_exc()}")
    finally:
        emit("exit")
