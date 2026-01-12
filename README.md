# KKBOX 情境式播放清單探勘與歌手分析儀表板

本專案以 **KKBOX Open API** 為資料來源，透過「情境關鍵字 → 播放清單 → 曲目歌手」的方式蒐集資料，並提供：  
1) 歌手在不同情境下的代表性（Lift/支持度門檻）計算  
2) 情境/歌手關鍵字探勘與文字雲  
3) PyQt5 桌面儀表板，支援查詢、快取、分群視覺化與圖片輸出

> 適用情境：資料探勘/推薦系統課程專題、情境式音樂分析、播放清單關鍵字與歌手關聯探索。

---

## 1. 專案結構

建議目錄如下（可依需求調整）：

```
.
├─ config.py                         # KKBOX API Client ID/Secret（請勿公開）
├─ kkbox_client.py                   # 建立 KKBOXAPI 的 helper
├─ crawl_kkbox.py                    # 依情境關鍵字爬取：context × playlist × artist
├─ crawl_playlist_meta_only.py       # 補抓 playlist metadata（title/description/image_url…）
├─ compute_lift.py                   # 計算 artist × context Lift（含 support/confidence 門檻）
├─ artist_keywords.py                # 歌手關鍵字抽取/文字雲（供 dashboard 使用）
├─ visualize.py                      # 熱力圖/文字雲等輸出（輸出到 outputs/）
├─ dashboard.py                      # PyQt5 儀表板（互動查詢、分群、輸出）
├─ requirements.txt
├─ data/
│  ├─ kkbox_raw_data.csv             # 爬蟲輸出（必要）
│  ├─ kkbox_playlist_meta.csv        # playlist meta（建議）
│  └─ qt_dashboard_state.pkl         # dashboard 狀態/快取（自動生成）
└─ outputs/
   ├─ cluster_latest.png             # dashboard 輸出的最新分群圖
   ├─ wordcloud_xx.png	       # 六情境代表性歌手文字雲
   └─                            
```

---

## 2. 環境需求

- Python 3.9+
- Windows 10/11
---

## 3. 安裝

### 3.1 建立虛擬環境並安裝套件

python -m pip install -r requirements.txt
---

## 4. KKBOX API 設定

### 4.1 設定 `config.py`

本專案以 `config.py` 提供 KKBOX Open API 的 `CLIENT_ID` / `CLIENT_SECRET`：

```python
# config.py
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
```

---

## 5. 資料產生流程（建議順序）

### Step A：依情境關鍵字爬取 raw data

直接執行`crawl_kkbox.py` ，但需要自行申請API金鑰故不建議重新爬取

>內建範例 `context_queries`可自行增刪關鍵字與情境，

```bash
python crawl_kkbox.py
```

輸出：
- `data/kkbox_raw_data.csv`（必要）  
  欄位：`context, search_term, playlist_id, artist`

> 若 API 呼叫過快可提高 `sleep_sec`，避免觸發限制。

---

### Step B（可選）：補抓 playlist metadata（不重爬 raw data）

若你已經有 `data/kkbox_raw_data.csv`，可用 `crawl_playlist_meta_only.py` 補抓完整 playlist meta：

```bash
python crawl_playlist_meta_only.py --raw data/kkbox_raw_data.csv --out data/kkbox_playlist_meta.csv
```

常用參數：
- `--sleep`：每次 API 呼叫後等待秒數（預設 0.35）
- `--save-every`：每抓 N 筆存檔一次（預設 50）
- `--limit`：只抓前 N 筆（測試用）

---

### Step C：計算 artist × context 的 Lift（含支持度門檻）

直接執行：
```bash
python compute_lift.py
```

此檔案目前是**在程式內**調整門檻：
- `MIN_PAIR_COUNT`：n_AC 最小共現次數（避免小樣本 Lift 虛高）
- `MIN_ARTIST_COUNT`：n_A 最小歌手次數
- `MIN_CONTEXT_COUNT`：n_C 最小情境次數
- `MIN_CONFIDENCE`：P(context|artist) 最小值（不想用可改 `None`）

輸出：
- `data/cross_tab_artist_context.csv`
- `data/lift_matrix_artist_context.csv`
- `data/lift_support_mask.csv`
- `data/top_artists_per_context.csv`

---

### Step D（可選）：產生靜態視覺化圖表

`visualize.py` 提供：
- 情境相似度熱力圖（由 Lift 矩陣推得）
- 情境文字雲（由播放清單 title/description 統計）

執行：
```bash
python visualize.py
```

輸出到：
- `outputs/heatmap_context_similarity.png`
- `outputs/wordcloud_<context>.png`

---

### Step E：啟動 PyQt5 儀表板（互動查詢/分群/輸出）

```bash
python dashboard.py
```

儀表板功能概覽：
- 以歌手為中心的情境雷達圖/文字雲
- Top-K 類似歌手與分群視覺化（PCA 2D 投影 + 階層式分群）
- 關閉程式時自動匯出分群圖到 `outputs/`：
  - `outputs/cluster_latest.png`
  - `outputs/cluster_pca_hier_<N>artists_<timestamp>.png`

---

## 6. 常見問題（Troubleshooting）

### 中文/日文/韓文字變方塊或缺字（Matplotlib / WordCloud）

- `dashboard.py` 已嘗試自動選擇 Windows 上可用的 CJK 字型（如 Microsoft JhengHei）。  
- `visualize.py` 的 WordCloud 會挑選常見 CJK 字型檔（例如 Noto/SourceHan 或 msjh.ttc）。

若你不是 Windows 或字型路徑不同：  
- 請在 `visualize.py` 的 `plot_context_wordcloud(font_path=...)` 明確指定字型檔（ttf/ttc/otf）。
