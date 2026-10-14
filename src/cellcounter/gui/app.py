"""CellCounter GUI (PySide6)"""
from __future__ import annotations

import json
import multiprocessing as mp
import queue as queue_mod
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import __version__
from ..config import Settings, user_config_path
from . import worker

COLUMNS = [
    ("FileName", "ファイル"), ("CellCount", "細胞数"), ("FilteredCellCount", "境界除去後"),
    ("EdgeRemovedCount", "除去数"), ("MeanArea_px", "平均面積(px)"), ("Seconds", "秒"), ("Error", "エラー"),
]


def _dspin(lo, hi, decimals, step) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setDecimals(decimals)
    w.setRange(lo, hi)
    w.setSingleStep(step)
    return w


def _ispin(lo, hi) -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    return w


class ImageView(QLabel):
    """ウィンドウサイズに合わせて縮小表示。ダブルクリックで既定のビューアで開く。"""

    def __init__(self):
        super().__init__("結果を選択するとオーバーレイを表示します\n（緑 = 採用 / 赤 = 境界除去）")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(200, 200)
        self.setStyleSheet("QLabel { background: #1e1e1e; color: #aaa; }")
        self._pix: QPixmap | None = None
        self._path = ""

    def show_file(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            self._pix, self._path = None, ""
            self.setPixmap(QPixmap())
            self.setText("オーバーレイ画像がありません")
            return
        self._pix, self._path = pix, path
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        self._rescale()
        super().resizeEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self._path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._path))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"CellCounter {__version__}")
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        self.ctx = mp.get_context("spawn")
        self.queue = self.ctx.Queue()
        self.proc = None
        self.stop_event = None
        self.env_info: dict = {}
        self.running = False
        self.out_dir: Path | None = None

        self._build_ui()
        self._apply_settings(self._load_user_settings())

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(100)

        self.log(f"CellCounter {__version__} 起動")
        self._start_env_check()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # ---- 左：設定パネル
        panel = QWidget()
        pv = QVBoxLayout(panel)

        g_io = QGroupBox("フォルダ")
        f = QFormLayout(g_io)
        self.ed_in, row_in = self._path_row(self._browse_input)
        self.ed_out, row_out = self._path_row(self._browse_output)
        self.ed_out.setPlaceholderText("空欄 = 画像フォルダ/results")
        f.addRow("画像", row_in)
        f.addRow("出力", row_out)
        pv.addWidget(g_io)

        g_model = QGroupBox("モデル / 推論")
        fm = QFormLayout(g_model)
        self.ed_model, row_model = self._path_row(self._browse_model, clear=True)
        self.ed_model.setPlaceholderText("空欄 = Cellpose デフォルトモデル")
        fm.addRow("モデル", row_model)
        self.cb_gpu = QCheckBox("GPU を使う")
        self.lb_device = QLabel("環境チェック中…")
        self.lb_device.setStyleSheet("color: gray")
        gpu_row = QHBoxLayout()
        gpu_row.addWidget(self.cb_gpu)
        gpu_row.addWidget(self.lb_device, 1)
        fm.addRow(gpu_row)
        self.sp_diam = _dspin(0, 2000, 1, 1)
        self.sp_diam.setSpecialValueText("自動")
        self.sp_diam.setToolTip("細胞の直径 (px)。0 = 自動（既定）")
        self.sp_flow = _dspin(0, 3, 2, 0.05)
        self.sp_flow.setToolTip("大きいほど検出数が増える（形の悪いマスクも許容）。既定 0.4")
        self.sp_prob = _dspin(-6, 6, 1, 0.5)
        self.sp_prob.setToolTip("小さいほど暗い細胞も検出（範囲 -6〜6）。既定 0.0")
        self.sp_min = _ispin(0, 100000)
        self.sp_min.setToolTip("これより小さい（px 数）マスクは除去。既定 15")
        fm.addRow("直径 (px)", self.sp_diam)
        fm.addRow("flow_threshold", self.sp_flow)
        fm.addRow("cellprob_threshold", self.sp_prob)
        fm.addRow("最小サイズ (px)", self.sp_min)
        b_param_reset = QPushButton("推論パラメータを既定値に戻す")
        b_param_reset.setToolTip("直径・flow_threshold・cellprob_threshold・最小サイズを既定値に戻します")
        b_param_reset.clicked.connect(self._reset_inference_params)
        prr = QHBoxLayout()
        prr.addStretch()
        prr.addWidget(b_param_reset)
        fm.addRow(prr)
        pv.addWidget(g_model)

        self.g_edge = QGroupBox("境界細胞除去")
        self.g_edge.setCheckable(True)
        ge = QGridLayout(self.g_edge)
        self.cb_top, self.cb_bottom = QCheckBox("上"), QCheckBox("下")
        self.cb_left, self.cb_right = QCheckBox("左"), QCheckBox("右")
        frame = QLabel("画像")
        frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame.setFixedSize(64, 44)
        frame.setStyleSheet("border: 1px solid gray; color: gray")
        ge.addWidget(self.cb_top, 0, 1, Qt.AlignmentFlag.AlignCenter)
        ge.addWidget(self.cb_left, 1, 0, Qt.AlignmentFlag.AlignRight)
        ge.addWidget(frame, 1, 1, Qt.AlignmentFlag.AlignCenter)
        ge.addWidget(self.cb_right, 1, 2, Qt.AlignmentFlag.AlignLeft)
        ge.addWidget(self.cb_bottom, 2, 1, Qt.AlignmentFlag.AlignCenter)
        self.sp_margin = _ispin(1, 500)
        self.sp_margin.setToolTip("端からこの距離 (px) 以内にかかる細胞を除去。Cellpose は端の 1〜2px 内側に輪郭を作るため 2 推奨")
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("マージン (px)"))
        mrow.addWidget(self.sp_margin)
        mrow.addStretch()
        ge.addLayout(mrow, 3, 0, 1, 3)
        pv.addWidget(self.g_edge)

        g_out = QGroupBox("出力")
        go = QVBoxLayout(g_out)
        self.cb_overlay = QCheckBox("輪郭オーバーレイ画像 (_overlay.png)")
        self.cb_table = QCheckBox("細胞ごとの計測 (cell_measurements.csv)")
        self.cb_masks = QCheckBox("マスク画像 (_cp_masks.png)")
        self.cb_seg = QCheckBox("seg.npy（Cellpose GUI で開ける）")
        for w in (self.cb_overlay, self.cb_table, self.cb_masks, self.cb_seg):
            go.addWidget(w)
        self.cmb_bg = QComboBox()
        self.cmb_bg.addItem("グレースケール", "gray")
        self.cmb_bg.addItem("元の色", "color")
        self.cmb_bg.setToolTip("位相差・明視野はグレースケール推奨。蛍光など色に意味がある画像は「元の色」")
        orow = QHBoxLayout()
        orow.setContentsMargins(22, 0, 0, 0)
        orow.addWidget(QLabel("背景"))
        orow.addWidget(self.cmb_bg, 1)
        go.insertLayout(1, orow)
        self.cb_overlay.toggled.connect(self.cmb_bg.setEnabled)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("ピクセルサイズ (µm/px)"))
        self.sp_px = _dspin(0, 1000, 4, 0.01)
        self.sp_px.setSpecialValueText("未設定")
        prow.addWidget(self.sp_px)
        go.addLayout(prow)
        pv.addWidget(g_out)

        brow = QHBoxLayout()
        b_load = QPushButton("設定を読み込む…")
        b_load.setToolTip("以前の出力フォルダの settings.json から設定を復元")
        b_load.clicked.connect(self._load_settings_file)
        b_reset = QPushButton("すべて初期値に戻す")
        b_reset.clicked.connect(lambda: self._apply_settings(Settings(
            input_dir=self.ed_in.text(), output_dir=self.ed_out.text())))
        brow.addWidget(b_load)
        brow.addWidget(b_reset)
        pv.addLayout(brow)
        pv.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(460)

        # ---- 右：結果
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_select)

        self.preview = ImageView()

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.table)
        top.addWidget(self.preview)
        top.setSizes([480, 420])

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(top)
        right.addWidget(self.log_view)
        right.setSizes([560, 180])

        # ---- 下：操作
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m")
        self.lb_status = QLabel("")
        self.btn_start = QPushButton("開始")
        self.btn_start.setEnabled(False)
        self.btn_start.setMinimumWidth(110)
        self.btn_start.setStyleSheet("QPushButton:enabled { font-weight: bold; }")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_open = QPushButton("出力フォルダを開く")
        self.btn_open.clicked.connect(self._open_output)
        self.btn_recheck = QPushButton("環境を再チェック")
        self.btn_recheck.clicked.connect(self._start_env_check)

        bottom = QHBoxLayout()
        bottom.addWidget(self.btn_recheck)
        bottom.addWidget(self.progress, 1)
        bottom.addWidget(self.lb_status)
        bottom.addWidget(self.btn_open)
        bottom.addWidget(self.btn_stop)
        bottom.addWidget(self.btn_start)

        body = QHBoxLayout()
        body.addWidget(scroll)
        body.addWidget(right, 1)

        central = QWidget()
        root = QVBoxLayout(central)
        root.addLayout(body, 1)
        root.addLayout(bottom)
        self.setCentralWidget(central)

    def _path_row(self, browse, clear=False):
        ed = QLineEdit()
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(ed, 1)
        b = QPushButton("参照…")
        b.clicked.connect(browse)
        h.addWidget(b)
        if clear:
            c = QPushButton("×")
            c.setFixedWidth(28)
            c.setToolTip("クリア（デフォルトモデルを使う）")
            c.clicked.connect(ed.clear)
            h.addWidget(c)
        return ed, w

    # ------------------------------------------------------------------ settings
    def _collect_settings(self) -> Settings:
        edges = [n for n, cb in (("top", self.cb_top), ("bottom", self.cb_bottom),
                                 ("left", self.cb_left), ("right", self.cb_right)) if cb.isChecked()]
        return Settings(
            input_dir=self.ed_in.text().strip(), output_dir=self.ed_out.text().strip(),
            model_path=self.ed_model.text().strip(), use_gpu=self.cb_gpu.isChecked(),
            diameter=self.sp_diam.value(), flow_threshold=self.sp_flow.value(),
            cellprob_threshold=self.sp_prob.value(), min_size=self.sp_min.value(),
            edge_filter=self.g_edge.isChecked(), edges=edges, edge_margin=self.sp_margin.value(),
            save_overlay=self.cb_overlay.isChecked(), overlay_background=self.cmb_bg.currentData(),
            save_masks=self.cb_masks.isChecked(),
            save_seg=self.cb_seg.isChecked(), save_cell_table=self.cb_table.isChecked(),
            pixel_size_um=self.sp_px.value(),
        )

    def _apply_settings(self, s: Settings):
        self.ed_in.setText(s.input_dir)
        self.ed_out.setText(s.output_dir)
        self.ed_model.setText(s.model_path)
        self.cb_gpu.setChecked(s.use_gpu)
        self.sp_diam.setValue(s.diameter)
        self.sp_flow.setValue(s.flow_threshold)
        self.sp_prob.setValue(s.cellprob_threshold)
        self.sp_min.setValue(s.min_size)
        self.g_edge.setChecked(s.edge_filter)
        for n, cb in (("top", self.cb_top), ("bottom", self.cb_bottom),
                      ("left", self.cb_left), ("right", self.cb_right)):
            cb.setChecked(n in s.edges)
        self.sp_margin.setValue(max(1, s.edge_margin))
        self.cb_overlay.setChecked(s.save_overlay)
        self.cmb_bg.setCurrentIndex(max(0, self.cmb_bg.findData(s.overlay_background)))
        self.cmb_bg.setEnabled(s.save_overlay)
        self.cb_masks.setChecked(s.save_masks)
        self.cb_seg.setChecked(s.save_seg)
        self.cb_table.setChecked(s.save_cell_table)
        self.sp_px.setValue(s.pixel_size_um)

    def _reset_inference_params(self):
        d = Settings()  # 既定値は config.Settings の 1 か所で管理
        self.sp_diam.setValue(d.diameter)
        self.sp_flow.setValue(d.flow_threshold)
        self.sp_prob.setValue(d.cellprob_threshold)
        self.sp_min.setValue(d.min_size)
        self.log(f"推論パラメータを既定値に戻しました（直径 自動, flow_threshold {d.flow_threshold}, "
                 f"cellprob_threshold {d.cellprob_threshold}, 最小サイズ {d.min_size}）")

    def _load_user_settings(self) -> Settings:
        path = user_config_path()
        try:
            return Settings.load(path) if path and path.is_file() else Settings()
        except Exception:
            return Settings()

    def _save_user_settings(self):
        path = user_config_path()
        if path is None:
            return
        try:
            self._collect_settings().save(path)
        except Exception:
            pass

    def _load_settings_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "settings.json を選択", self.ed_out.text() or self.ed_in.text(),
                                              "JSON (*.json)")
        if not path:
            return
        try:
            s = Settings.load(Path(path))
        except Exception as e:
            QMessageBox.warning(self, "読み込み失敗", str(e))
            return
        # フォルダは今の指定を優先（別データに同じ条件を適用する用途が多いため）
        s.input_dir = self.ed_in.text() or s.input_dir
        s.output_dir = self.ed_out.text()
        self._apply_settings(s)
        self.log(f"設定を読み込みました: {path}")

    # ------------------------------------------------------------------ browse / drop
    def _browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "画像フォルダを選択", self.ed_in.text())
        if d:
            self.ed_in.setText(d)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "出力フォルダを選択", self.ed_out.text() or self.ed_in.text())
        if d:
            self.ed_out.setText(d)

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Cellpose モデルを選択", self.ed_model.text(), "すべてのファイル (*)")
        if p:
            self.ed_model.setText(p)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                self.ed_in.setText(str(p))
                self.log(f"画像フォルダ: {p}")
                break

    def _open_output(self):
        s = self._collect_settings()
        d = self.out_dir or (s.resolved_output_dir() if s.input_dir else None)
        if d and Path(d).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    # ------------------------------------------------------------------ env check
    def _start_env_check(self):
        if self.running:
            return
        self.btn_start.setEnabled(False)
        self.btn_recheck.setEnabled(False)
        self.lb_device.setText("環境チェック中…")
        self.log("Python / Cellpose 環境をチェック中…")
        p = self.ctx.Process(target=worker.env_check_entry, args=(self.queue,), daemon=True)
        p.start()

    def _on_env(self, info: dict):
        self.env_info = info
        self.btn_recheck.setEnabled(True)
        self.log(f"Python {info.get('python')}  ({info.get('python_executable')})")
        if not info.get("cellpose_import_ok"):
            self.lb_device.setText("Cellpose が使えません")
            self.log(f"エラー: {info.get('error')}")
            self.log("→ この Python に `pip install cellpose` を実行してから「環境を再チェック」してください")
            return
        self.log(f"Cellpose {info.get('cellpose')} / PyTorch {info.get('torch')} / {info.get('device_name')}")
        gpu = bool(info.get("gpu_available"))
        self.cb_gpu.setEnabled(gpu)
        if not gpu:
            self.cb_gpu.setChecked(False)
            self.log("GPU が利用できないため CPU で実行します")
        self.lb_device.setText(info.get("device_name", ""))
        self.lb_device.setStyleSheet("color: green" if gpu else "color: gray")
        self.btn_start.setEnabled(True)
        self.log("環境チェック完了")

    # ------------------------------------------------------------------ batch
    def _start(self):
        s = self._collect_settings()
        if not s.input_dir or not Path(s.input_dir).is_dir():
            QMessageBox.warning(self, "CellCounter", "画像フォルダを指定してください")
            return
        if s.edge_filter and not s.edges:
            QMessageBox.warning(self, "CellCounter", "境界細胞除去の方向を 1 つ以上選んでください")
            return
        if s.model_path and not Path(s.model_path).exists():
            QMessageBox.warning(self, "CellCounter", f"モデルファイルが見つかりません:\n{s.model_path}")
            return
        if not self.cb_gpu.isEnabled():
            s.use_gpu = False
        self._save_user_settings()

        self.out_dir = s.resolved_output_dir()
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self.stop_event = self.ctx.Event()
        self.proc = self.ctx.Process(
            target=worker.batch_entry, args=(s.to_dict(), self.env_info, self.queue, self.stop_event), daemon=True)
        self.proc.start()
        self._set_running(True)
        self.lb_status.setText("モデル読み込み中…")

    def _stop(self):
        if not self.running:
            return
        if self.stop_event is not None and not self.stop_event.is_set():
            self.stop_event.set()
            self.log("停止要求：現在の画像の処理が終わり次第止めます（もう一度押すと強制停止）")
            self.btn_stop.setText("強制停止")
            return
        self._terminate()

    def _terminate(self):
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(3)
            self.log("強制停止しました（処理済みの画像は image_summary.csv に保存されています）")
            self._mark_record("terminated")
        self._set_running(False)

    def _mark_record(self, status: str):
        if not self.out_dir:
            return
        p = self.out_dir / "settings.json"
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            rec.update(status=status, finished=datetime.now().isoformat(timespec="seconds"))
            p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _set_running(self, running: bool):
        self.running = running
        self.btn_start.setEnabled(not running and bool(self.env_info.get("cellpose_import_ok")))
        self.btn_stop.setEnabled(running)
        self.btn_stop.setText("停止")
        self.btn_recheck.setEnabled(not running)
        if not running:
            self.proc = None
            self.stop_event = None

    # ------------------------------------------------------------------ events
    def _poll(self):
        for _ in range(200):
            try:
                kind, data = self.queue.get_nowait()
            except queue_mod.Empty:
                break
            self._handle(kind, data)
        # 異常終了（メモリ不足等）でワーカーが消えた場合
        if self.running and self.proc is not None and not self.proc.is_alive() and self.queue.empty():
            code = self.proc.exitcode
            if code not in (0, None):
                self.log(f"解析プロセスが異常終了しました（終了コード {code}）")
                self._mark_record("crashed")
            self._set_running(False)

    def _handle(self, kind: str, d: dict):
        if kind == "env":
            self._on_env(d["info"])
        elif kind == "log":
            self.log(d["message"])
        elif kind == "start":
            self.progress.setMaximum(max(1, d["total"]))
            self.progress.setValue(0)
            self.log(f"バッチ処理開始（{d['total']} 枚）")
        elif kind == "result":
            self._add_row(d["row"])
        elif kind == "progress":
            self.progress.setValue(d["done"])
            self.lb_status.setText(f"{d['done']}/{d['total']}")
        elif kind == "finished":
            sm = d["summary"]
            msg = {"completed": "完了", "cancelled": "停止"}.get(sm["status"], sm["status"])
            self.log(f"{msg}: {sm['processed']}/{sm['total']} 枚, エラー {sm['errors']} 件, "
                     f"細胞 {sm['total_cells']} 個（境界除去後 {sm['total_filtered_cells']} 個）, {sm['seconds']} 秒")
            self.log(f"結果: {sm['summary_csv']}")
            self.lb_status.setText(msg)
        elif kind == "fatal":
            self.log(f"エラー: {d['message']}")
            self.lb_status.setText("エラー")
            QMessageBox.critical(self, "CellCounter", d["message"].splitlines()[0])
        elif kind == "exit":
            if self.proc is not None:
                self.proc.join(5)
            self._set_running(False)

    def _add_row(self, row: dict):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, (key, _) in enumerate(COLUMNS):
            v = row.get(key, "")
            item = QTableWidgetItem()
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                item.setData(Qt.ItemDataRole.DisplayRole, v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                item.setText(str(v))
            if c == 0:
                item.setData(Qt.ItemDataRole.UserRole, row.get("overlay", ""))
            if row.get("Error"):
                item.setForeground(QBrush(QColor("#d32f2f")))
            self.table.setItem(r, c, item)
        # 最新の結果を自動表示（ユーザーが別の行を見ていない場合）
        sel = self.table.selectionModel().selectedRows()
        if not sel or sel[0].row() == r - 1:
            self.table.selectRow(r)
        self.table.scrollToItem(self.table.item(r, 0))

    def _on_select(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            path = self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
            self.preview.show_file(path or "")

    def log(self, msg: str):
        self.log_view.appendPlainText(f"{datetime.now():%H:%M:%S}  {msg}")

    def closeEvent(self, e):
        if self.running:
            ans = QMessageBox.question(self, "CellCounter", "解析中です。中止して終了しますか？")
            if ans != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self._terminate()
        self._save_user_settings()
        super().closeEvent(e)


def main() -> int:
    mp.freeze_support()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CellCounter")
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
