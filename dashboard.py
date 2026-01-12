# qt_dashboard.py
# -*- coding: utf-8 -*-
import sys
import traceback
from io import BytesIO
import numpy as np
import pandas as pd
import re
import urllib.request
from typing import Optional
import os
import pickle
from datetime import datetime
import html

from PyQt5.QtCore import Qt, QTimer, QRunnable, QThreadPool, pyqtSignal, QObject, QPoint, QSize
from PyQt5.QtGui import QPixmap, QFont, QPainter, QPainterPath, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QFrame, QSizePolicy, QMenu, QAction, QPushButton,
    QListWidget, QListWidgetItem, QToolButton, QAbstractItemView,
    QScrollArea, QCompleter, QGroupBox
)

# Matplotlib (Radar chart)
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from kkbox_client import get_kkbox_api
from artist_keywords import extract_keywords_from_titles, generate_wordcloud_image

from collections import defaultdict,Counter

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =========================
# 0) Fonts: Qt + Matplotlib CJK
# =========================

def configure_matplotlib_cjk_font():
    """
    Force Matplotlib to use a CJK-capable font on Windows.
    Primary: Microsoft JhengHei (微軟正黑體)
    Fallbacks: Microsoft YaHei, Noto Sans CJK, SimHei, Arial Unicode MS
    """
    from matplotlib import font_manager as fm

    candidates = [
        "Microsoft JhengHei",
        "Microsoft JhengHei UI",
        "Microsoft YaHei",
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "SimHei",
        "Arial Unicode MS",
        "Malgun Gothic",
        "Malgun Gothic Semilight",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]

    found = None
    for name in candidates:
        try:
            # Try to resolve; will fallback if not found. We want a real match.
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            if path:
                found = name
                break
        except Exception:
            continue

    if found is None:
        # If none found, still set to candidates list; Matplotlib may resolve something.
        found = "Microsoft JhengHei"

    matplotlib.rcParams["font.sans-serif"] = [found] + candidates
    matplotlib.rcParams["axes.unicode_minus"] = False


# =========================
# 1) Data logic (ported from app.py)
# =========================

RAW_PATH = "data/kkbox_raw_data.csv"
REQUIRED_COLS = ["context", "search_term", "playlist_id", "artist"]

CONTEXT_ORDER = ['工作/讀書', '失戀/分手', '告白/求婚', '運動', '派對/狂歡', '通勤/開車']

playlist_title_cache = {}  # playlist_id -> (title, description)

META_PATH = "data/kkbox_playlist_meta.csv"

LOCAL_STATE_PATH = "data/qt_dashboard_state.pkl"
CLUSTER_EXPORT_DIR = "outputs"
# playlist_id -> (title, description)
playlist_meta_map = {}

def load_playlist_meta(path: str):
    try:
        m = pd.read_csv(path)
        if "playlist_id" not in m.columns:
            return {}

        m["playlist_id"] = m["playlist_id"].astype(str).str.strip()

        if "title" not in m.columns:
            m["title"] = ""
        if "description" not in m.columns:
            m["description"] = ""

        m["title"] = m["title"].astype(str)
        m["description"] = m["description"].astype(str)

        return {
            row["playlist_id"]: (row.get("title", "") or "", row.get("description", "") or "")
            for _, row in m.iterrows()
        }
    except Exception:
        return {}


def load_raw_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"kkbox_raw_data.csv 缺少欄位: {missing}")

    df["context"] = df["context"].astype(str).str.strip()
    df["playlist_id"] = df["playlist_id"].astype(str).str.strip()
    df["artist"] = df["artist"].astype(str).str.strip()
    df = df.dropna(subset=["context", "playlist_id", "artist"])
    df = df.drop_duplicates()
    return df


def get_artist_context_stats(df_raw: pd.DataFrame, artist_name: str, max_playlists_per_artist: int = 200):
    """
    回傳：
      playlist_ids: list[str]
      context_counts: dict[context->count]
    """
    df_a = df_raw[df_raw["artist"] == artist_name].copy()
    if df_a.empty:
        return [], {ctx: 0 for ctx in CONTEXT_ORDER}

    g = df_a.groupby("context")["playlist_id"].nunique()
    context_counts = {ctx: int(g.get(ctx, 0)) for ctx in CONTEXT_ORDER}

    playlist_ids = (
        df_a["playlist_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if len(playlist_ids) > max_playlists_per_artist:
        playlist_ids = playlist_ids[:max_playlists_per_artist]

    return playlist_ids, context_counts


def get_playlist_title_desc(kkbox, playlist_id: str):
    playlist_id = str(playlist_id).strip()
    if not playlist_id:
        return "", ""

    # memory cache
    if playlist_id in playlist_title_cache:
        return playlist_title_cache[playlist_id]

    # ✅ local meta cache
    if playlist_id in playlist_meta_map:
        playlist_title_cache[playlist_id] = playlist_meta_map[playlist_id]
        return playlist_title_cache[playlist_id]

    # fallback: KKBOX API
    try:
        data = kkbox.shared_playlist_fetcher.fetch_shared_playlist(playlist_id, terr="TW")
        title = (data.get("title") or "").strip()
        desc = (data.get("description") or "").strip()
        playlist_title_cache[playlist_id] = (title, desc)
        return title, desc
    except Exception:
        playlist_title_cache[playlist_id] = ("", "")
        return "", ""


# =========================
# 2) Thread helpers (avoid UI blocking)
# =========================

class WorkerSignals(QObject):
    finished = pyqtSignal(object)   # result
    error = pyqtSignal(str)         # traceback string


class FnWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(res)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


# =========================
# 3) Widgets: Radar chart + Main UI
# =========================

class RadarCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, polar=True)

    def plot(self, labels, values, title: str):
        self.ax.clear()

        if not labels:
            labels = CONTEXT_ORDER
            values = [0] * len(labels)

        n = len(labels)
        import math
        angles = [2 * math.pi * i / n for i in range(n)]
        angles += angles[:1]
        vals = list(values) + values[:1]

        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(labels, fontsize=14)     
        self.ax.tick_params(axis='y', labelsize=12)      
        self.ax.tick_params(axis='x', labelsize=14)

        vmax = max(vals) if vals else 0
        self.ax.set_ylim(0, max(1, vmax))

        self.ax.plot(angles, vals, linewidth=2)
        self.ax.fill(angles, vals, alpha=0.25)

        self.ax.set_title(title, pad=22, fontsize=16)   
        # 每次 plot 後重新做 layout
        try:
            self.fig.tight_layout(rect=[0.04, 0.04, 0.96, 0.90], pad=1.2)
        except Exception:
            self.fig.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.88)

        self.draw()

class ScatterCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 3.8), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

    def plot(self, coords, labels, names, title: str):
        self.ax.clear()

        if coords is None or len(coords) == 0:
            self.ax.set_title(title)
            self.ax.text(0.5, 0.5, "尚未加入足夠歌手（至少 2 位）", ha="center", va="center")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.draw()
            return

        x = coords[:, 0]
        y = coords[:, 1]
        self.ax.scatter(x, y, c=labels)

        for i, nm in enumerate(names):
            self.ax.text(x[i] + 0.01, y[i] + 0.01, nm, fontsize=10)

        self.ax.set_xlabel("PCA 1")
        self.ax.set_ylabel("PCA 2")
        self.ax.set_title(title)
        self.fig.tight_layout()
        self.draw()

class CacheRowWidget(QWidget):
    def __init__(self, artist_name: str, on_pick, on_add, on_remove_cluster, on_clear, parent=None):
        super().__init__(parent)
        self.artist_name = artist_name

        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 2, 6, 2)
        outer.setSpacing(0)

        left = QWidget(self)
        left_lay = QHBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)  # 你若想更貼近可改 4

        self.btn_name = QPushButton(artist_name, left)
        self.btn_name.setFlat(True)
        self.btn_name.setCursor(Qt.PointingHandCursor)
        self.btn_name.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.btn_name.setStyleSheet("text-align: left; padding-left: 2px;")
        self.btn_name.clicked.connect(lambda: on_pick(self.artist_name))
        left_lay.addWidget(self.btn_name)

        self.btn_add = QToolButton(left)
        self.btn_add.setText("+")
        self.btn_add.setFixedSize(26, 24)
        self.btn_add.setCursor(Qt.PointingHandCursor)
        self.btn_add.setToolTip("加入此歌手到分群圖")
        self.btn_add.clicked.connect(lambda: on_add(self.artist_name))
        left_lay.addWidget(self.btn_add)

        # x：只負責「從分群圖移除」
        self.btn_remove = QToolButton(left)
        self.btn_remove.setText("x")
        self.btn_remove.setFixedSize(26, 24)
        self.btn_remove.setCursor(Qt.PointingHandCursor)
        self.btn_remove.setToolTip("從分群圖移除")
        self.btn_remove.clicked.connect(lambda: on_remove_cluster(self.artist_name))
        left_lay.addWidget(self.btn_remove)

        # 清除：清除「該歌手」快取 + 分群
        self.btn_clear = QToolButton(left)
        self.btn_clear.setText("清除")
        self.btn_clear.setFixedHeight(24)
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.setToolTip("清除該歌手快取，並同時從分群移除")
        self.btn_clear.clicked.connect(lambda: on_clear(self.artist_name))
        left_lay.addWidget(self.btn_clear)

        outer.addWidget(left, 0)
        outer.addStretch(1)

    def set_in_cluster(self, in_cluster: bool):
        # +：未加入才可按
        self.btn_add.setEnabled(not in_cluster)
        self.btn_add.setToolTip("已加入分群" if in_cluster else "加入此歌手到分群圖")

        # x：已加入才可按
        self.btn_remove.setEnabled(in_cluster)
        self.btn_remove.setToolTip("從分群圖移除" if in_cluster else "尚未加入分群")

        # 清除：永遠可按
        self.btn_clear.setEnabled(True)

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("歌手情境分析儀表板")
        self.resize(1450, 1200)

        # state
        self.threadpool = QThreadPool.globalInstance()
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(300)
        self.debounce.timeout.connect(self._do_search)

        self.last_suggestions = []  # list[dict{id,name}]
        self.kkbox = None
        self.df_raw = None

        # suggestion menu (QCompleter popup)
        self.suggest_model = QStandardItemModel(self)
        self.completer = QCompleter(self.suggest_model, self)
        self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)  # 模糊包含
        self.completer.activated.connect(self._on_completer_activated)

        # UI
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content = QWidget()
        scroll.setWidget(content)

        self.setCentralWidget(scroll)
        self._build_ui(content)
        
        self.current_artist_name = ""       # 目前已選定分析的歌手（用於 + 按鈕）
        self.current_artist_id = ""
        self.current_artist_image_url = ""
        self.cluster_artists = []           # 已加入分群視覺化的歌手列表（保持順序、去重）
        self.artist_cache = {}   # artist_name -> res dict
        self.cache_rows = {}     # artist_name -> (QListWidgetItem, CacheRowWidget)
        self._artist_req_token = 0
        self._image_url_cache = {}   # url -> bytes

        
        # init resources
        self._init_data()
        self._load_local_state()


    def _build_ui(self, parent: QWidget):
        root = QVBoxLayout(parent)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        title = QLabel("歌手情境分析儀表板")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        root.addWidget(title)

        subtitle = QLabel("輸入歌手名稱後，系統會即時查 KKBOX 候選歌手；選定後將產生情境屬性雷達圖並以生成關鍵字文字雲。")
        subtitle.setStyleSheet("color: #555;")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # Search bar row
        search_row = QHBoxLayout()
        root.addLayout(search_row)

        self.input = QLineEdit()
        self.input.setPlaceholderText("請輸入歌手名稱，例如：五月天、周杰倫")
        self.input.textChanged.connect(self._on_text_changed)
        self.input.returnPressed.connect(self._on_enter)
        self.input.setMinimumHeight(34)
        self.input.setCompleter(self.completer)
        search_row.addWidget(self.input, 1)
        
        self.status = QLabel("尚未選擇歌手。")
        self.status.setStyleSheet("color: #444; margin-top: 2px;")
        root.addWidget(self.status)

        # --- artist + cache (same row) ---
        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        root.addLayout(top_row)

        # left: artist box (image + name in ONE outer frame)
        self.artist_box = QFrame()
        self.artist_box.setObjectName("artistBox")
        self.artist_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.artist_box.setFixedWidth(300)  # 可依你 UI 調整
        self.artist_box.setStyleSheet("""
        QFrame#artistBox {
            border: 1px solid #cfcfcf;
            border-radius: 10px;
            background: white;
        }
        """)
        top_row.addWidget(self.artist_box, 0)

        box_lay = QVBoxLayout(self.artist_box)
        box_lay.setContentsMargins(10, 10, 10, 10)
        box_lay.setSpacing(8)
        box_lay.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        self.artist_img = QLabel()
        self.artist_img.setFixedSize(144, 144)
        self.artist_img.setAlignment(Qt.AlignCenter)
        # ✅ 不要再給 artist_img 自己的 frame，避免雙框
        self.artist_img.setStyleSheet("background: transparent;")
        box_lay.addWidget(self.artist_img, alignment=Qt.AlignHCenter)

        self.artist_name = QLabel("（尚未選擇歌手）")
        self.artist_name.setStyleSheet("font:bold 32px; font-weight: 600;")
        self.artist_name.setWordWrap(True)
        self.artist_name.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.artist_name.setMinimumHeight(90)
        box_lay.addWidget(self.artist_name)

        # right: cache list block
        right_col = QVBoxLayout()
        right_col.setSpacing(3)
        top_row.addLayout(right_col, 1)

        cache_title = QLabel("快取清單（點歌手可重新顯示雷達圖/文字雲；名稱右側可透過按鈕加入分群或移除）")
        cache_title.setStyleSheet("font-weight: 600;")
        right_col.addWidget(cache_title)

        self.cache_list = QListWidget()
        self.cache_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.cache_list.setMinimumHeight(80)  # 右側空間更大，可自行調
        self.cache_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 不要水平卷軸
        self.cache_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        right_col.addWidget(self.cache_list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        right_col.addLayout(btn_row)

        # Panels
        panels = QHBoxLayout()
        panels.setSpacing(12)
        root.addLayout(panels, 1)
        # --- separator ---
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        # --- clustering section ---
        cluster_title = QLabel("歌手分群視覺化（6情境比例向量 → Hierarchical，PCA 2D投影）")
        cluster_title.setStyleSheet("font-weight: 600;")
        root.addWidget(cluster_title)

        cluster_row = QHBoxLayout()
        root.addLayout(cluster_row, 0)

        # left: scatter plot
        self.scatter_canvas = ScatterCanvas(parent)
        self.scatter_canvas.setMinimumHeight(600)
        self.scatter_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        cluster_row.addWidget(self.scatter_canvas, 3)

 # right: summary
        right_box = QVBoxLayout()
        cluster_row.addLayout(right_box, 1)

        # ===== Top-3 cosine nearest neighbors =====
        self.neighbor_group = QGroupBox("Cluster中最相似的其他歌手")
        self.neighbor_group.setStyleSheet("QGroupBox { font-weight: 600; }")
        ng_layout = QVBoxLayout(self.neighbor_group)

        self.neighbor_status = QLabel("Top-3：請先選定歌手並加入分群後才會顯示。")
        self.neighbor_status.setWordWrap(True)
        self.neighbor_status.setTextFormat(Qt.RichText)
        self.neighbor_status.setFrameShape(QFrame.StyledPanel)
        self.neighbor_status.setMinimumHeight(200)
        self.neighbor_status.setStyleSheet("font-size: 24px;")
        ng_layout.addWidget(self.neighbor_status)

        right_box.addWidget(self.neighbor_group)

        # ===== original cluster summary =====
        self.cluster_status = QLabel("尚未加入歌手。請在「快取清單」每列右側按「+」加入分群。")
        self.cluster_status.setWordWrap(True)
        self.cluster_status.setFrameShape(QFrame.StyledPanel)
        self.cluster_status.setMinimumHeight(500)
        right_box.addWidget(self.cluster_status)

        # initial empty plot
        self.scatter_canvas.plot(None, None, None, "Hierarchical分群（尚未加入足夠歌手）")
        # Radar panel
        radar_panel = QVBoxLayout()
        panels.addLayout(radar_panel, 1)

        radar_title = QLabel("六大情境歌單屬性分佈雷達圖")
        radar_title.setStyleSheet("font-weight: 600;")
        radar_panel.addWidget(radar_title)

        # Radar stacked area: (0) canvas, (1) loading label
        self.radar_area = QWidget()
        self.radar_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.radar_layout = QVBoxLayout(self.radar_area)
        self.radar_layout.setContentsMargins(0, 0, 0, 0)

        self.radar_canvas = RadarCanvas(self.radar_area)
        self.radar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.radar_canvas.setMinimumSize(500, 500)   # 視窗夠大時更舒服，可自行調
        # 不要 setMaximumSize

        self.radar_loading = QLabel("請先選擇一位歌手。")
        self.radar_loading.setAlignment(Qt.AlignCenter)
        self.radar_loading.setFrameShape(QFrame.StyledPanel)

        # ✅ 直接放進 layout，不要用 stretch + 置中
        self.radar_layout.addWidget(self.radar_canvas)
        self.radar_layout.addWidget(self.radar_loading)

        self.radar_canvas.show()
        self.radar_loading.hide()


        radar_panel.addWidget(self.radar_area, 1)

        # Wordcloud panel
        wc_panel = QVBoxLayout()
        panels.addLayout(wc_panel, 1)

        wc_title = QLabel("歌單關鍵字文字雲")
        wc_title.setStyleSheet("font-weight: 600;")
        wc_panel.addWidget(wc_title)

        self.wc_area = QWidget()
        self.wc_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.wc_layout = QVBoxLayout(self.wc_area)
        self.wc_layout.setContentsMargins(0, 0, 0, 0)

        self.wordcloud = QLabel("請先選擇一位歌手以生成文字雲。")
        self.wordcloud.setAlignment(Qt.AlignCenter)
        self.wordcloud.setFrameShape(QFrame.StyledPanel)
        self.wordcloud.setMinimumHeight(360)
        self.wordcloud.setScaledContents(False)

        self.wc_loading = QLabel("等待輸入…")
        self.wc_loading.setAlignment(Qt.AlignCenter)
        self.wc_loading.setFrameShape(QFrame.StyledPanel)

        self.wc_layout.addWidget(self.wordcloud)
        self.wc_layout.addWidget(self.wc_loading)
        self.wordcloud.show()
        self.wc_loading.hide()

        wc_panel.addWidget(self.wc_area, 1)

        # initial empty radar
        self.radar_canvas.plot(CONTEXT_ORDER, [0] * 6, "（尚未選擇歌手）")

    def _init_data(self):
        self.df_raw = load_raw_df(RAW_PATH)

        global playlist_meta_map
        playlist_meta_map = load_playlist_meta(META_PATH)

        self.kkbox = get_kkbox_api()


    # -------------------------
    # Loading hints
    # -------------------------
    def _set_radar_hint(self, text: str, show: bool):
        if show:
            self.radar_loading.setText(text)
            self.radar_canvas.hide()
            self.radar_loading.show()
        else:
            self.radar_loading.hide()
            self.radar_canvas.show()

    def _set_wc_hint(self, text: str, show: bool):
        if show:
            self.wc_loading.setText(text)
            self.wordcloud.hide()
            self.wc_loading.show()
        else:
            self.wc_loading.hide()
            self.wordcloud.show()

    # -------------------------
    # Search / suggestions (QMenu popup)
    # -------------------------
    def _on_text_changed(self, _):
        q = self.input.text().strip()
        if not q:
            self.last_suggestions = []
            self.suggest_model.clear()
            self.completer.popup().hide()
            return
        self.debounce.start()


    def _do_search(self):
        q = self.input.text().strip()
        if not q:
            return

        self.status.setText("正在向 KKBOX 查詢候選歌手…")
        self.completer.popup().hide()

        worker = FnWorker(self._kkbox_search_artists, q)
        worker.signals.finished.connect(self._on_search_done)
        worker.signals.error.connect(self._on_worker_error)
        self.threadpool.start(worker)


    def _kkbox_search_artists(self, q: str):
        result = self.kkbox.search_fetcher.search(q, types=["artist"], terr="TW")
        artists = result.get("artists", {}).get("data", [])

        out = []
        for a in artists:
            images = a.get("images") or []
            # 取第一個可用 URL（多數情況 images[0] 是最大張）
            img_url = ""
            for im in images:
                u = (im.get("url") or "").strip()
                if u:
                    img_url = u
                    break

            out.append({
                "id": a.get("id"),
                "name": a.get("name"),
                "image_url": img_url,
            })
        return out


    def _on_search_done(self, artists):
        self.last_suggestions = artists or []
        self.suggest_model.clear()

        if not self.last_suggestions:
            self.status.setText("查無候選歌手。")
            self.completer.popup().hide()
            return

        for a in self.last_suggestions[:15]:
            name = (a.get("name") or "").strip()
            if not name:
                continue
            it = QStandardItem(name)
            it.setData(a, Qt.UserRole)  # 把整包資料塞進去（含 id / image_url）
            self.suggest_model.appendRow(it)

        # ✅ 顯示候選清單，但游標仍留在輸入框（QCompleter 的特性）
        self.completer.complete()
        self.status.setText("請從候選歌手清單中點選一位載入分析。")


    def _pick_suggestion(self, a: dict):
        name = (a.get("name") or "").strip()
        aid = (a.get("id") or "").strip()
        img_url = (a.get("image_url") or "").strip()
        if not name:
            return

        self.debounce.stop()
        self.completer.popup().hide()
        self.last_suggestions = []

        self.current_artist_name = name
        self.current_artist_id = aid
        self.current_artist_image_url = img_url

        # 先顯示名字，圖片等分析回來再放
        self._set_artist_header(name, None)
        self._load_artist(name, aid, img_url)
        QTimer.singleShot(0, self._clear_search_input)


        


    def _on_enter(self):
        self.debounce.stop()
        self.completer.popup().hide()

        if self.last_suggestions:
            a = self.last_suggestions[0]
            name = (a.get("name") or "").strip()
            aid = (a.get("id") or "").strip()
            if name:
                self.last_suggestions = []

                self.input.blockSignals(True)
                self.input.setText(name)
                self.input.blockSignals(False)

                img_url = (a.get("image_url") or "").strip()
                self._load_artist(name, aid, img_url)

                return

        name = self.input.text().strip()
        if name:
            self.last_suggestions = []
            self._load_artist(name, "", "")


    # -------------------------
    # Load artist dashboard
    # -------------------------
    def _load_artist(self, artist_name: str, artist_id: str, artist_image_url: str):
        self.current_artist_name = artist_name
        self.current_artist_id = artist_id

        # token：避免多次點擊/切換造成舊圖片覆蓋新圖片
        self._artist_req_token += 1
        token = self._artist_req_token

        # 先顯示名字（圖片先清空），接著立刻非阻塞預取圖片
        self._set_artist_header(artist_name, None)
        self._start_artist_image_prefetch(artist_name, artist_image_url, token)

        self.status.setText(f"正在分析「{artist_name}」…")
        self._set_radar_hint("正在分析情境雷達圖…", True)
        self._set_wc_hint("正在產生文字雲…", True)

        worker = FnWorker(self._build_artist_outputs, artist_name, artist_id, artist_image_url)
        worker.signals.finished.connect(lambda res: self._on_artist_done(artist_name, res))
        worker.signals.error.connect(self._on_worker_error)
        self.threadpool.start(worker)


    def _build_artist_outputs(self, artist_name: str, artist_id: str, artist_image_url: str):

        # 1) local CSV stats
        used_name = artist_name
        playlist_ids, context_counts = get_artist_context_stats(self.df_raw, artist_name)

        # ✅ fallback：raw_data 找不到 → 省略括號內容再找一次
        if not playlist_ids:
            alt = self._strip_parentheses(artist_name)
            if alt and alt != artist_name:
                playlist_ids2, context_counts2 = get_artist_context_stats(self.df_raw, alt)
                if playlist_ids2:
                    used_name = alt
                    playlist_ids, context_counts = playlist_ids2, context_counts2

        labels = list(context_counts.keys())
        counts = [context_counts[c] for c in labels]

        # 2) wordcloud via KKBOX playlist titles + descriptions
        wc_png_bytes = None
        if playlist_ids:
            texts = []

            def _clean_text(s) -> str:
                if s is None:
                    return ""
                s = str(s).strip()
                # pandas NaN 轉字串後常見為 'nan'
                if s.lower() == "nan":
                    return ""
                return s

            for pid in playlist_ids:
                title, desc = get_playlist_title_desc(self.kkbox, pid)
                title = _clean_text(title)
                desc  = _clean_text(desc)

                # ✅ 文字雲：同時使用「標題 + 描述」
                combined = (title + " " + desc).strip()
                if combined:
                    texts.append(combined)

            if texts:
                counter = extract_keywords_from_titles(texts, used_name)
                if counter:
                    img_bytesio = generate_wordcloud_image(counter)  # expected BytesIO
                    if img_bytesio is not None:
                        if isinstance(img_bytesio, BytesIO):
                            wc_png_bytes = img_bytesio.getvalue()
                        else:
                            wc_png_bytes = img_bytesio

        artist_image_bytes = None
        url = (artist_image_url or "").strip()
        if url:
            # ✅ 若主執行緒已預取成功，這裡直接重用
            if hasattr(self, "_image_url_cache") and url in self._image_url_cache:
                artist_image_bytes = self._image_url_cache.get(url)
            else:
                try:
                    with urllib.request.urlopen(url, timeout=8) as resp:
                        artist_image_bytes = resp.read()
                    # 背景也可寫入 cache，避免下次再抓
                    if hasattr(self, "_image_url_cache") and artist_image_bytes:
                        self._image_url_cache[url] = artist_image_bytes
                except Exception:
                    artist_image_bytes = None


        return {
            "labels": labels,
            "counts": counts,
            "wc_png_bytes": wc_png_bytes,
            "has_playlists": bool(playlist_ids),
            "artist_image_bytes": artist_image_bytes,
            "display_artist_name": used_name,
        }


    def _on_artist_done(self, artist_name: str, res: dict):
        show_name = (res.get("display_artist_name") or artist_name).strip()
        labels = res.get("labels") or CONTEXT_ORDER
        counts = res.get("counts") or [0] * len(labels)

        # radar
        self.radar_canvas.plot(labels, counts, show_name)
        self._set_radar_hint("", False)
        self._set_artist_header(show_name, res.get("artist_image_bytes"))
        # wordcloud
        wc_bytes = res.get("wc_png_bytes")
        wc_ok = False
        if res.get("artist_image_bytes"):
            self._set_artist_header(show_name, res.get("artist_image_bytes"))

        if wc_bytes:
            pix = QPixmap()
            wc_ok = pix.loadFromData(wc_bytes, "PNG")
            if wc_ok:
                self.wordcloud.setPixmap(pix)
                self.wordcloud.setAlignment(Qt.AlignCenter)
            else:
                self.wordcloud.setPixmap(QPixmap())
                self.wordcloud.setText("文字雲載入失敗（PNG bytes 異常）。")
        else:
            self.wordcloud.setPixmap(QPixmap())
            if not res.get("has_playlists"):
                self.wordcloud.setText("找不到該歌手的歌單紀錄，無法產生文字雲。")
            else:
                self.wordcloud.setText("無法產生文字雲")

        self._set_wc_hint("", False)

        # 快取條件：必須在 raw_data 有資料（has_playlists=True）
        if res.get("has_playlists"):
            # 目前用 self.current_artist_id 當 artist_id 來源
            self._cache_upsert(show_name, self.current_artist_id, res)
            self.status.setText(f"完成「{artist_name}」的情境分析（已加入快取）。")
        else:
            self.status.setText(f"完成「{artist_name}」的情境分析（raw_data 無資料，不加入快取）。")


    def _on_worker_error(self, tb: str):
        self.completer.popup().hide()
        self._set_radar_hint("發生錯誤，無法繪製雷達圖。", True)
        self._set_wc_hint("發生錯誤，無法產生文字雲。", True)
        self.status.setText("發生錯誤，請查看終端機輸出。")
        print(tb)
    
    def _strip_parentheses(self, name: str) -> str:
        s = re.sub(r"\s*[\(（][^)\）]*[\)）]\s*", " ", str(name))
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _make_display_names(self, names):
        base = [self._strip_parentheses(n) for n in names]
        seen = defaultdict(int)
        out = []
        for b in base:
            seen[b] += 1
            out.append(b if seen[b] == 1 else f"{b}#{seen[b]}")
        return out

    def _artist_profile_ratio(self, artist_name: str):
        """
        回傳 (counts_list, ratio_list) 皆為 CONTEXT_ORDER 長度的 list。
        """
        _, context_counts = get_artist_context_stats(self.df_raw, artist_name)
        counts = [context_counts[c] for c in CONTEXT_ORDER]
        s = sum(counts)
        if s <= 0:
            ratio = [0.0] * len(counts)
        else:
            ratio = [c / s for c in counts]
        return counts, ratio

    def _refresh_cluster_view(self):
        """
        Hierarchical (Agglomerative) clustering on 6D context ratio vectors:
        - metric: cosine
        - linkage: average
        - k: auto-chosen by silhouette (2..min(10, n-1))
        PCA(2) is ONLY for visualization.

        Uses:
        - self.cluster_artists: list[str]
        - self.scatter_canvas: ScatterCanvas (matplotlib canvas with .fig/.ax)
        - self.cluster_status: QLabel (summary)
        """
        # ---------- guard ----------
        if not hasattr(self, "scatter_canvas"):
            return
        if not hasattr(self, "cluster_artists"):
            self.cluster_artists = []

        # de-duplicate while preserving order
        seen = set()
        artists = []
        for a in self.cluster_artists:
            a = (a or "").strip()
            if not a or a in seen:
                continue
            seen.add(a)
            artists.append(a)

        canvas = self.scatter_canvas
        fig = getattr(canvas, "fig", None)
        ax = getattr(canvas, "ax", None)
        if fig is None:
            fig = getattr(canvas, "figure", None)
        if ax is None and fig is not None and fig.axes:
            ax = fig.axes[0]
        if ax is None:
            return

        ax.clear()

        def _set_neighbor_status(text: str):
            if hasattr(self, "neighbor_status"):
                self.neighbor_status.setText(text)

        def _strip_parentheses(name: str) -> str:
            s = re.sub(r"\s*[\(（][^)\）]*[\)）]\s*", " ", str(name))
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _make_display_names(names):
            base = [_strip_parentheses(n) for n in names]
            cnt = Counter(base)
            seen2 = Counter()
            out = []
            for b in base:
                if cnt[b] == 1:
                    out.append(b)
                else:
                    seen2[b] += 1
                    out.append(f"{b}#{seen2[b]}")
            return out

        # ---------- not enough ----------
        if len(artists) < 2:
            ax.set_title("Hierarchical 分群（至少需 2 位歌手）", fontsize=12)
            ax.text(0.5, 0.5, "尚未加入足夠歌手（至少 2 位）", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            if hasattr(self, "cluster_status"):
                self.cluster_status.setText("尚未加入足夠歌手（至少 2 位）。")
            _set_neighbor_status("Top-3 cosine 最近鄰：cluster需要至少 2 位歌手，且需先選定歌手後才可計算。")
            canvas.draw()
            return

        # ---------- build ratio vectors from local CSV ----------
        df = self.df_raw
        df_sub = df[df["artist"].isin(artists)].copy()

        if df_sub.empty:
            ax.set_title("Hierarchical 分群（raw_data 無紀錄）", fontsize=12)
            ax.text(0.5, 0.5, "raw_data 中找不到已加入歌手的紀錄", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            if hasattr(self, "cluster_status"):
                self.cluster_status.setText("raw_data 中找不到已加入歌手的紀錄，無法分群。")
            _set_neighbor_status("Top-3 cosine 最近鄰：無有效紀錄，無法計算。")
            canvas.draw()
            return

        g = (
            df_sub.groupby(["artist", "context"])["playlist_id"]
            .nunique()
            .unstack(fill_value=0)
        )

        # ensure context columns
        for ctx in CONTEXT_ORDER:
            if ctx not in g.columns:
                g[ctx] = 0

        # keep the order of artists (and only those present)
        present = [a for a in artists if a in g.index]
        g = g.loc[present, CONTEXT_ORDER].copy()

        if g.shape[0] < 2:
            ax.set_title("Hierarchical 分群（有效歌手不足）", fontsize=12)
            ax.text(0.5, 0.5, "有效歌手不足（至少 2 位需在 raw_data 中有紀錄）", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            if hasattr(self, "cluster_status"):
                self.cluster_status.setText("有效歌手不足（至少 2 位需在 raw_data 中有紀錄）。")
            _set_neighbor_status("Top-3 cosine 最近鄰：有效歌手不足，無法計算。")
            canvas.draw()
            return

        sums = g.sum(axis=1).replace(0, 1)
        X = (g.div(sums, axis=0)).values.astype(float)
        names = list(g.index)

        # ---------- auto choose k via silhouette (cosine) ----------
        n = X.shape[0]
        if n == 2:
            best_k = 2
            best_s = None
        else:
            kmax = min(10, n - 1)
            best_k, best_s = 2, -1.0

            for k in range(2, kmax + 1):
                try:
                    model = AgglomerativeClustering(
                        n_clusters=k,
                        linkage="average",
                        metric="cosine",
                    )
                    labels = model.fit_predict(X)
                except TypeError:
                    model = AgglomerativeClustering(
                        n_clusters=k,
                        linkage="average",
                        affinity="cosine",
                    )
                    labels = model.fit_predict(X)

                if len(set(labels)) < 2:
                    continue

                s = silhouette_score(X, labels, metric="cosine")
                if s > best_s:
                    best_s, best_k = s, k

        # final clustering
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=best_k,
                linkage="average",
                metric="cosine",
            )
            y = clusterer.fit_predict(X)
        except TypeError:
            clusterer = AgglomerativeClustering(
                n_clusters=best_k,
                linkage="average",
                affinity="cosine",
            )
            y = clusterer.fit_predict(X)

        # ---------- PCA for 2D visualization ----------
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(X)

        ax.scatter(coords[:, 0], coords[:, 1], c=y)

        # 決定：當前歌手要標紅哪一個點（若不在分群圖內就不標）
        highlight_idx = None
        cur = (getattr(self, "current_artist_name", "") or "").strip()
        if cur:
            # 1) 先用原字串
            if cur in names:
                highlight_idx = names.index(cur)
            else:
                # 2) 再用去括號」後的名字比對
                cur_base = _strip_parentheses(cur)
                bases = [_strip_parentheses(nm) for nm in names]
                if cur_base in bases:
                    # 若有重名，取第一個
                    highlight_idx = bases.index(cur_base)

        

        disp_names = _make_display_names(names)

        # ===== Top-3 cosine nearest neighbors (6D X under cosine) =====
        cur = (getattr(self, "current_artist_name", "") or "").strip()

        if not cur:
            _set_neighbor_status("Current: (未選定)\nTop-3：請先選定一位歌手。")
        elif highlight_idx is None:
            _set_neighbor_status(f"Current: {cur}\nTop-3：該歌手不在分群圖內（請先加入分群）。")
        else:
            cur_disp = disp_names[highlight_idx]
            cur_lab = int(y[highlight_idx])

            v = X[highlight_idx]
            vn = float(np.linalg.norm(v))
            if vn <= 0:
                _set_neighbor_status(f"Current: {cur_disp} | Cluster {cur_lab}\nTop-3：向量全為 0，無法計算。")
            else:
                sims = []
                for j in range(n):
                    if j == highlight_idx:
                        continue
                    u = X[j]
                    un = float(np.linalg.norm(u))
                    if un <= 0:
                        sim = -1.0
                    else:
                        sim = float(np.dot(v, u) / (vn * un))
                    sims.append((sim, j))

                sims.sort(key=lambda t: t[0], reverse=True)
                top = sims[:3]
                cur_disp_html = html.escape(cur_disp)
                header = f"<b>Current:</b> <b>{cur_disp_html}</b> | Cluster {cur_lab}"

                items_html = []
                for rank, (sim, j) in enumerate(top, 1):
                    name_html = html.escape(disp_names[j])
                    lab_j = int(y[j])
                    items_html.append(
                        f"{rank}) <b>{name_html}</b><br>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;cosine={sim:.3f} | Cluster {lab_j}"
                    )

                body = "<br>".join(items_html)
                out = (
                    f"{header}<br><br>"
                    f"Top-3:</b><br>"
                    f"{body}"
                )

                _set_neighbor_status(out)
        for i, nm in enumerate(disp_names):
            kw = {}
            if highlight_idx is not None and i == highlight_idx:
                kw["color"] = "red"
                kw["fontweight"] = "bold"
            ax.text(coords[i, 0] -0.005, coords[i, 1] + 0.005, nm, fontsize=14, **kw)


        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")

        if n == 2:
            title = "Hierarchical 分群（k=2；cosine+average）"
            status = "Hierarchical（cosine+average），n=2 → k=2（silhouette 不適用）"
        else:
            title = f"Hierarchical 分群（k={best_k}；cosine+average）"
            status = f"Hierarchical（cosine+average），n={n}，auto k={best_k}，silhouette={best_s:.3f}"

        ax.set_title(title, fontsize=12)

        # summary (cluster members)
        if hasattr(self, "cluster_status"):
            groups = defaultdict(list)
            for nm, lab in zip(names, y):
                groups[int(lab)].append(nm)

            lines = [status, ""]
            for lab in sorted(groups.keys()):
                lines.append(f"Cluster {lab}:")
                for nm in groups[lab]:
                    lines.append(f"  - {nm}")
                lines.append("")
            self.cluster_status.setText("\n".join(lines).rstrip())

        fig.tight_layout()
        canvas.draw()
    
    def _cache_upsert(self, artist_name: str, artist_id: str, res: dict, persist: bool = True):

        """
        將分析結果寫入快取，並在快取清單建立一列（歌手名）。
        """
        artist_name = (artist_name or "").strip()
        if not artist_name:
            return

        if not hasattr(self, "artist_cache"):
            self.artist_cache = {}
        if not hasattr(self, "cache_rows"):
            self.cache_rows = {}

        res2 = dict(res or {})
        res2["artist_id"] = (artist_id or res2.get("artist_id", "") or "")
        self.artist_cache[artist_name] = res2

        # 建立 UI 列（僅第一次）
        if artist_name not in self.cache_rows:
            item = QListWidgetItem()
            self.cache_list.addItem(item)

            row = CacheRowWidget(
                artist_name=artist_name,
                on_pick=self._load_from_cache,
                on_add=self._add_artist_to_cluster_from_cache,
                on_remove_cluster=self._remove_artist_from_cluster,  # x：移出分群
                on_clear=self._clear_artist,                         # 清除：刪快取 + 移出分群
                parent=self.cache_list,
            )

            item.setSizeHint(row.sizeHint())
            self.cache_list.setItemWidget(item, row)
            self._fix_cache_row_widths()

            self.cache_rows[artist_name] = (item, row)

        # 同步「已加入分群」狀態，決定該列 + 是否 disabled
        self._refresh_cache_row_states()
        if persist:
            self._save_local_state()

            

    def _refresh_cache_row_states(self):
        cluster_set = set(self.cluster_artists or [])
        for name, (_item, row) in self.cache_rows.items():
            row.set_in_cluster(name in cluster_set)

    def _load_from_cache(self, artist_name: str):
        artist_name = (artist_name or "").strip()
        if artist_name not in self.artist_cache:
            self.status.setText("快取不存在或已被移除。")
            return

        res = self.artist_cache[artist_name]
        labels = res.get("labels") or CONTEXT_ORDER
        counts = res.get("counts") or [0] * len(labels)

        # 避免觸發 textChanged → debounce → QMenu
        self.debounce.stop()
        self.completer.popup().hide()
        self.last_suggestions = []

        self.current_artist_name = artist_name
        self.current_artist_id = res.get("artist_id", "")
        self._set_artist_header(artist_name, res.get("artist_image_bytes"))
        # radar
        self.radar_canvas.plot(labels, counts, artist_name)
        self._set_radar_hint("", False)

        # wordcloud
        wc_bytes = res.get("wc_png_bytes")
        if wc_bytes:
            pix = QPixmap()
            if pix.loadFromData(wc_bytes, "PNG"):
                self.wordcloud.setPixmap(pix)
                self.wordcloud.setAlignment(Qt.AlignCenter)
        else:
            self.wordcloud.setPixmap(QPixmap())
            if not res.get("has_playlists"):
                self.wordcloud.setText("找不到該歌手的歌單紀錄，無法產生文字雲。")
            else:
                self.wordcloud.setText("無法產生文字雲。")

        self._set_wc_hint("", False)
        self._refresh_cluster_view()
        self.status.setText(f"已從快取載入「{artist_name}」。")

    def _add_artist_to_cluster_from_cache(self, artist_name: str):
        name = (artist_name or "").strip()
        if not name:
            return

        if name not in self.cluster_artists:
            self.cluster_artists.append(name)
            self.status.setText(f"已加入「{name}」到分群視覺化。")
            self._refresh_cluster_view()
        else:
            self.status.setText(f"「{name}」已在分群清單中。")

        # 更新每列 + disabled 狀態
        self._refresh_cache_row_states()
        self._save_local_state()


    def _delete_from_cache(self, artist_name: str):
        name = (artist_name or "").strip()
        if not name:
            return

        # 只刪快取；不自動移除分群（避免你分群圖突然少點）
        self.artist_cache.pop(name, None)

        if name in self.cache_rows:
            item, _row = self.cache_rows.pop(name)
            r = self.cache_list.row(item)
            if r >= 0:
                self.cache_list.takeItem(r)

        self.status.setText(f"已從快取清單移除「{name}」。")
        self._save_local_state()
    
    def _fix_cache_row_widths(self):
        if not hasattr(self, "cache_list"):
            return
        w = self.cache_list.viewport().width()
        for i in range(self.cache_list.count()):
            item = self.cache_list.item(i)
            row = self.cache_list.itemWidget(item)
            if row is None:
                continue
            item.setSizeHint(QSize(w, row.sizeHint().height()))
    
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fix_cache_row_widths()
    def _to_round_pixmap(self, pix: QPixmap, size: int) -> QPixmap:
        """
        將 pix 裁切成圓形，輸出 size x size（透明背景）。
        """
        if pix.isNull():
            out = QPixmap(size, size)
            out.fill(Qt.transparent)
            return out

        # 先裁成正方形並填滿
        src = pix.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

        out = QPixmap(size, size)
        out.fill(Qt.transparent)

        p = QPainter(out)
        p.setRenderHint(QPainter.Antialiasing, True)

        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        p.setClipPath(path)

        # 將圖貼上（已是填滿的正方形）
        p.drawPixmap(0, 0, src)
        p.end()
        return out

    def _set_artist_header(self, name: str, img_bytes):
        self.artist_name.setText(name or "（尚未選擇歌手）")

        size = self.artist_img.width()  
        self.artist_img.setPixmap(QPixmap())

        if img_bytes:
            pix = QPixmap()
            if pix.loadFromData(img_bytes):
                self.artist_img.setPixmap(self._to_round_pixmap(pix, size))
                return

        # 沒有圖：顯示空白圓框
        placeholder = QPixmap(size, size)
        placeholder.fill(Qt.transparent)
        self.artist_img.setPixmap(placeholder)
    
    def _fetch_image_bytes(self, url: str):
        url = (url or "").strip()
        if not url:
            return None
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                return resp.read()
        except Exception:
            return None

    def _start_artist_image_prefetch(self, artist_name: str, image_url: str, token: int):
        """
        非阻塞預取圖片：若已在 cache 直接更新；否則開 worker 下載後再更新 header。
        """
        url = (image_url or "").strip()
        if not url:
            return

        # 已有快取：直接顯示
        if url in self._image_url_cache:
            if token == self._artist_req_token and artist_name == self.current_artist_name:
                self._set_artist_header(artist_name, self._image_url_cache.get(url))
            return

        worker = FnWorker(self._fetch_image_bytes, url)
        worker.signals.finished.connect(lambda b, n=artist_name, u=url, t=token: self._on_artist_image_prefetched(n, u, t, b))
        worker.signals.error.connect(self._on_worker_error)
        self.threadpool.start(worker)

    def _on_artist_image_prefetched(self, artist_name: str, url: str, token: int, img_bytes):
        if img_bytes:
            self._image_url_cache[url] = img_bytes

        # 只更新「目前仍在看的那位」的 header，避免快速切換歌手
        if token != self._artist_req_token:
            return
        if artist_name != self.current_artist_name:
            return

        self._set_artist_header(artist_name, img_bytes)
    
    def _save_local_state(self):
        """
        將 artist_cache + cluster_artists 存到本地（pickle）。
        """
        try:
            os.makedirs(os.path.dirname(LOCAL_STATE_PATH) or ".", exist_ok=True)

            data = {
                "artist_cache": self.artist_cache or {},
                "cluster_artists": self.cluster_artists or [],
            }

            tmp = LOCAL_STATE_PATH + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, LOCAL_STATE_PATH)
        except Exception:
            print("[WARN] failed to save local state:")
            traceback.print_exc()


    def _load_local_state(self):
        """
        從本地載入 artist_cache + cluster_artists，並重建快取清單 UI。
        """
        if not os.path.exists(LOCAL_STATE_PATH):
            return

        try:
            with open(LOCAL_STATE_PATH, "rb") as f:
                data = pickle.load(f)

            self.artist_cache = data.get("artist_cache", {}) or {}
            self.cluster_artists = data.get("cluster_artists", []) or []

            # 重建快取清單 UI
            self.cache_list.clear()
            self.cache_rows = {}

            # 依插入順序還原
            for artist_name, res in self.artist_cache.items():
                # 只還原有效快取
                if not (res or {}).get("has_playlists", True):
                    continue
                self._cache_upsert(artist_name, (res or {}).get("artist_id", ""), res, persist=False)

            self._refresh_cache_row_states()

            # 還原分群視覺化
            self._refresh_cluster_view()
            self._refresh_cache_row_states()

            self.status.setText("已載入本地快取狀態。")
        except Exception:
            print("[WARN] failed to load local state:")
            traceback.print_exc()


    def closeEvent(self, e):
        # 關閉視窗前：先輸出分群圖，再存本地狀態
        self._export_cluster_plot()
        self._save_local_state()
        super().closeEvent(e)

    
    def _remove_artist_from_cluster(self, artist_name: str):
        name = (artist_name or "").strip()
        if not name:
            return

        if name in self.cluster_artists:
            self.cluster_artists = [a for a in self.cluster_artists if a != name]
            self.status.setText(f"已將「{name}」從分群圖移除。")
            self._refresh_cluster_view()
        else:
            self.status.setText(f"「{name}」尚未加入分群。")

        self._refresh_cache_row_states()
        if hasattr(self, "_save_local_state"):
            self._save_local_state()


    def _clear_artist(self, artist_name: str):
        """
        清除「單一歌手」：從快取移除 + 從分群移除 + 更新 UI + 落盤
        """
        name = (artist_name or "").strip()
        if not name:
            return

        # 1) 從分群移除
        if name in self.cluster_artists:
            self.cluster_artists = [a for a in self.cluster_artists if a != name]
            self._refresh_cluster_view()

        # 2) 刪快取資料
        self.artist_cache.pop(name, None)

        # 3) 刪除快取清單那一列
        if name in self.cache_rows:
            item, _row = self.cache_rows.pop(name)
            r = self.cache_list.row(item)
            if r >= 0:
                self.cache_list.takeItem(r)

        # 4) 若目前正在顯示該歌手，順便把畫面重置（避免顯示已不存在的快取）
        if getattr(self, "current_artist_name", "") == name:
            self.current_artist_name = ""
            self.current_artist_id = ""
            self.current_artist_image_url = ""
            self._set_artist_header("", None)
           
            self._set_radar_hint("請先選擇一位歌手。", True)
            self._set_wc_hint("等待輸入…", True)
            self.wordcloud.setPixmap(QPixmap())
            self.wordcloud.setText("請先選擇一位歌手以生成文字雲。")

        self._refresh_cache_row_states()
        if hasattr(self, "_save_local_state"):
            self._save_local_state()

        self.status.setText(f"已清除「{name}」的快取與分群狀態。")
    
    def _on_completer_activated(self, text: str):
        # 由顯示文字反查 item，取出 Qt.UserRole 內的 dict
        for r in range(self.suggest_model.rowCount()):
            it = self.suggest_model.item(r)
            if it and it.text() == text:
                a = it.data(Qt.UserRole)
                if isinstance(a, dict):
                    self._pick_suggestion(a)
                break

    def _clear_search_input(self):
        self.input.blockSignals(True)
        self.input.clear()
        self.input.setFocus()
        self.input.blockSignals(False)
    
    def _export_cluster_plot(self):
        """
        關閉程式時，把目前分群圖（PCA scatter）輸出成 PNG。
        會輸出兩份：
        1) cluster_latest.png（覆蓋，方便直接看最新）
        2) cluster_pca_hier_{N}artists_{timestamp}.png（保留歷史）
        """
        try:
            if not hasattr(self, "scatter_canvas") or self.scatter_canvas is None:
                return

            os.makedirs(CLUSTER_EXPORT_DIR, exist_ok=True)

            n = len(self.cluster_artists or [])
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            latest_path = os.path.join(CLUSTER_EXPORT_DIR, "cluster_latest.png")
            hist_path = os.path.join(CLUSTER_EXPORT_DIR, f"cluster_pca_hier_{n}artists_{ts}.png")

            self.scatter_canvas.fig.savefig(latest_path, dpi=200, bbox_inches="tight")
            self.scatter_canvas.fig.savefig(hist_path, dpi=200, bbox_inches="tight")

            print(f"[INFO] cluster plot exported: {latest_path}")
            print(f"[INFO] cluster plot exported: {hist_path}")
        except Exception:
            print("[WARN] failed to export cluster plot:")
            traceback.print_exc()




def main():
    configure_matplotlib_cjk_font()

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft JhengHei", 10))

    w = Dashboard()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
