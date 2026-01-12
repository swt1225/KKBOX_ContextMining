# compute_lift.py
#
# 計算「歌手 × 情境」Lift(提升度) 並加入 support 門檻（避免小樣本 Lift 虛高）。
#
# Lift(artist, context) = P(artist|context) / P(artist)
# 亦等價於：lift = (n_AC * N) / (n_A * n_C)
#
# 門檻（可調）：
#   - min_pair_count    : n_AC 最小共現次數
#   - min_artist_count  : n_A 最小歌手總次數
#   - min_context_count : n_C 最小情境總次數
#   - min_confidence    : P(context|artist)=n_AC/n_A 最小值（選配）
#
# 不符合門檻的格子，預設輸出 NaN（方便下游做 Top-K 排名時自動忽略）。

import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple



def is_chinese_name(name: str) -> bool:
    """
    視為『華語歌手』的規則：
    - 只看括號外的文字是否包含至少一個中文字
    - 括號可以是中括號（）、半形括號() 都會被處理
      例如：
        '周杰倫 (Jay Chou)'      -> True  （括號外有中文）
        'Taylor Swift (泰勒絲)'  -> False （中文只出現在括號內）
    """
    if not isinstance(name, str):
        return False

    # 移除所有括號及其中內容（中英文括號都處理）
    cleaned = re.sub(r'[\(\（][^)\）]*[\)\）]', '', name)

    # 在括號外檢查是否有中文字
    return re.search(r'[\u4e00-\u9fff]', cleaned) is not None


def load_and_clean_raw(path: str = "data/kkbox_raw_data.csv") -> pd.DataFrame:
    """
    讀取爬回來的原始資料，做基本清洗：
    - 去掉缺失值
    - 去掉空白
    - 去除完全重複列
    - 同一情境 + 同一歌單 + 同一歌手，只算一次
    """
    df = pd.read_csv(path)

    expected_cols = ["context", "search_term", "playlist_id", "artist"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"原始檔缺少欄位: {missing}")

    df = df[expected_cols]

    df = df.dropna(subset=["context", "playlist_id", "artist"])

    for col in expected_cols:
        df[col] = df[col].astype(str).str.strip()

    #df = df.drop_duplicates()

    # 同一情境 + 同一歌單 + 同一歌手，只算一次
    #df = df.drop_duplicates(subset=["context", "playlist_id", "artist"])

    return df


def build_cross_tab(df: pd.DataFrame) -> pd.DataFrame:
    """建立「歌手 × 情境」共現矩陣（count matrix）。"""
    return pd.crosstab(df["artist"], df["context"])


def compute_support_mask(
    cross_tab: pd.DataFrame,
    min_pair_count: int = 10,
    min_artist_count: int = 30,
    min_context_count: int = 0,
    min_confidence: Optional[float] = 0.05,
) -> pd.DataFrame:
    """
    回傳 bool mask（同形狀）表示哪些 (artist, context) 允許保留。

    - min_pair_count: n_AC
    - min_artist_count: n_A
    - min_context_count: n_C
    - min_confidence: P(C|A)=n_AC/n_A
    """
    if cross_tab.empty:
        return cross_tab.astype(bool)

    # n_AC
    mask = cross_tab >= int(min_pair_count)

    # n_A
    artist_totals = cross_tab.sum(axis=1)
    mask &= artist_totals.to_numpy()[:, None] >= int(min_artist_count)

    # n_C
    context_totals = cross_tab.sum(axis=0)
    mask &= context_totals.to_numpy()[None, :] >= int(min_context_count)

    # P(C|A)
    if min_confidence is not None:
        denom = artist_totals.replace(0, np.nan)
        conf = cross_tab.div(denom, axis=0)
        mask &= conf >= float(min_confidence)

    return mask


def compute_lift_matrix(
    cross_tab: pd.DataFrame,
    *,
    min_pair_count: int = 10,
    min_artist_count: int = 30,
    min_context_count: int = 0,
    min_confidence: Optional[float] = 0.05,
    invalid_to_nan: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    計算 Lift 矩陣並套用 support 門檻。

    回傳：
      - lift_matrix: DataFrame(artist x context)
      - mask       : DataFrame(bool) 同形狀，True 表示符合門檻

    invalid_to_nan:
      True  -> 不符合門檻設為 NaN（推薦；Top-K 排名會自動忽略）
      False -> 不符合門檻設為 0.0
    """
    if cross_tab.empty:
        return cross_tab.astype(float), cross_tab.astype(bool)

    # 先算 Lift（向量化）
    artist_totals = cross_tab.sum(axis=1)   # n_A
    context_totals = cross_tab.sum(axis=0)  # n_C
    N = float(artist_totals.sum())          # 總出現次數

    expected = np.outer(artist_totals.to_numpy(), context_totals.to_numpy()) / N
    lift = cross_tab.to_numpy(dtype=float) / expected
    lift = np.where(np.isfinite(lift), lift, np.nan)

    lift_matrix = pd.DataFrame(lift, index=cross_tab.index, columns=cross_tab.columns)

    # support mask
    mask = compute_support_mask(
        cross_tab,
        min_pair_count=min_pair_count,
        min_artist_count=min_artist_count,
        min_context_count=min_context_count,
        min_confidence=min_confidence,
    )

    if invalid_to_nan:
        lift_matrix = lift_matrix.where(mask, np.nan)
    else:
        lift_matrix = lift_matrix.where(mask, 0.0)

    return lift_matrix, mask


def export_top_artists_per_context(
    lift_matrix: pd.DataFrame,
    top_k: int = 20,
    out_path: str = "data/top_artists_per_context.csv",
):
    """輸出每個情境 Lift 最高的前 top_k 位歌手（自動忽略 NaN）。"""
    records = []
    for ctx in lift_matrix.columns:
        s = lift_matrix[ctx].dropna().sort_values(ascending=False).head(int(top_k))
        for artist, lift_val in s.items():
            records.append({"context": ctx, "artist": artist, "lift": float(lift_val)})

    df_top = pd.DataFrame(records)
    df_top.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已輸出每個情境的 Top {top_k} 歌手至 {out_path}")


if __name__ == "__main__":
    # ====== support 門檻（直接在這裡調整） ======
    MIN_PAIR_COUNT = 5       # n_AC
    MIN_ARTIST_COUNT = 25     # n_A
    MIN_CONTEXT_COUNT = 0     # n_C
    MIN_CONFIDENCE = 0.05     # P(C|A)（不想用就改 None）

    # 1) 讀取與清洗
    df_raw = load_and_clean_raw("data/kkbox_raw_data.csv")
    print("原始筆數（去重後 context+playlist+artist）：", len(df_raw))

    # 2) 建立共現矩陣
    cross_tab = build_cross_tab(df_raw)
    cross_tab.to_csv("data/cross_tab_artist_context.csv", encoding="utf-8-sig")
    print("共現矩陣形狀：", cross_tab.shape)

    # 3) 計算 Lift + 套用門檻
    lift_matrix, mask = compute_lift_matrix(
        cross_tab,
        min_pair_count=MIN_PAIR_COUNT,
        min_artist_count=MIN_ARTIST_COUNT,
        min_context_count=MIN_CONTEXT_COUNT,
        min_confidence=MIN_CONFIDENCE,
        invalid_to_nan=True,
    )

    lift_matrix.to_csv("data/lift_matrix_artist_context.csv", encoding="utf-8-sig")
    mask.to_csv("data/lift_support_mask.csv", encoding="utf-8-sig")

    kept = int(mask.to_numpy().sum())
    total_cells = int(mask.size)
    print(f"Lift 矩陣形狀： {lift_matrix.shape}，保留格子：{kept}/{total_cells} ({kept/total_cells:.2%})")

    # 4) 輸出每個情境的 Top 歌手列表（方便檢查）
    export_top_artists_per_context(
        lift_matrix,
        top_k=30,
        out_path="data/top_artists_per_context.csv",
    )

    print("Lift 計算流程完成")
