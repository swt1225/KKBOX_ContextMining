# visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from typing import Optional
import re
# 視覺化前先設定字體
plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",  # 中文(繁)
    "Malgun Gothic",       # 韓文
]
plt.rcParams["axes.unicode_minus"] = False  # 避免負號顯示為方塊


def plot_keyword_similarity_heatmap(lift_path: str = "data/lift_matrix_artist_context.csv",
                                    out_path: str = "outputs/heatmap_context_similarity.png"):
    """
    根據 Lift 矩陣計算情境之間的相關係數，畫出情境相似度熱力圖。
    用來觀察：哪些情境在「代表歌手組合」上是相近的。
    """
    lift_matrix = pd.read_csv(lift_path, index_col=0)

    # 若 lift_matrix 由 support 門檻產生 NaN（代表該 artist-context 配對資訊不足），
    # 直接做 corr() 可能導致大量 NaN 的相關係數。
    # 這裡用 0 補齊「未通過門檻的格子」以便比較情境之間在代表歌手分佈上的相似性。
    corr = lift_matrix.fillna(0).corr().fillna(0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr,
                annot=True,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                square=True)
    plt.title("情境關鍵字相似度（依歌手 Lift）")
    plt.tight_layout()

    # 確保目錄存在
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ 已輸出情境相似度熱力圖至 {out_path}")

def _pick_cjk_font_path() -> str:
    """Pick an installed font that covers CJK (Chinese/Japanese/Korean) for WordCloud."""
    candidates = [
        "C:/Windows/Fonts/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/SourceHanSansTC-Regular.otf",
        "C:/Windows/Fonts/SourceHanSans-Regular.otf",

        # 沒裝 Noto/SourceHan 才退而求其次（但可能缺字）
        "C:/Windows/Fonts/msjh.ttc",      # 繁中
        "C:/Windows/Fonts/msyh.ttc",      # 簡中
        "C:/Windows/Fonts/meiryo.ttc",    # 日文
        "C:/Windows/Fonts/msgothic.ttc",  # 日文
        "C:/Windows/Fonts/malgun.ttf",    # 韓文
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "找不到可用的 CJK 字型檔。請確認 C:/Windows/Fonts 內有日文/韓文字型，"
        "或自行在 plot_context_wordcloud(font_path=...) 指定字型路徑。"
    )
_HANGUL_RE = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uAC00-\uD7AF\uD7B0-\uD7FF]")

_PAREN_PATTERNS = [
    re.compile(r"\(([^()]*)\)"),      # ( ... )
    re.compile(r"（([^（）]*)）"),       # （ ... ） full-width
]

def _contains_hangul(s: str) -> bool:
    return bool(_HANGUL_RE.search(s or ""))

def normalize_artist_label(name: str) -> str:
    if not isinstance(name, str):
        return name
    out = name
    for pat in _PAREN_PATTERNS:
        def _repl(m):
            inner = m.group(1)
            return "" if _contains_hangul(inner) else m.group(0)
        out = pat.sub(_repl, out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out
def plot_context_wordcloud(context: str,
                           lift_path: str = "data/lift_matrix_artist_context.csv",
                           out_path: Optional[str] = None,
                           top_k: int = 20,
                           font_path: Optional[str] = None):
    """
    針對單一情境，取 Lift 最高的前 top_k 位歌手，畫代表歌手文字雲。
    - 字越大代表 Lift 越高：
      某歌手相對其他情境來說，特別代表這個情境。
    """
    lift_matrix = pd.read_csv(lift_path, index_col=0)

    if context not in lift_matrix.columns:
        raise ValueError(f"情境 '{context}' 不存在於 Lift 矩陣欄位中")

    # 由於 lift_matrix 可能含 NaN（support 門檻濾除），需要先清掉 NaN，
    # 否則 WordCloud 會在計算字體大小時拋出 "cannot convert float NaN to integer"。
    series = lift_matrix[context].dropna()
    series = series[series.replace([float("inf"), float("-inf")], pd.NA).notna()]
    series = series[series > 0]
    series = series.sort_values(ascending=False).head(top_k)
    series.index = series.index.map(normalize_artist_label)
    series = series[series.index.astype(str).str.len() > 0]
    series = series.groupby(level=0).max().sort_values(ascending=False).head(top_k)
    
    if series.empty:
        print(f"⚠️ 情境『{context}』在門檻條件下沒有足夠資料可生成文字雲（全部被濾除或為 NaN）。")
        return

    freq_dict = series.to_dict()

    if font_path is None:
        font_path = _pick_cjk_font_path()

    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white",
        prefer_horizontal=1.0
    )
    wc.generate_from_frequencies(freq_dict)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc)
    plt.axis("off")
    plt.title(f"情境「{context}」代表歌手 Lift 文字雲")
    plt.tight_layout()

    if out_path is None:
        safe_ctx = context.replace("/", "_")
        out_path = f"outputs/wordcloud_{safe_ctx}.png"

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ 已輸出情境「{context}」代表歌手文字雲至 {out_path}")


if __name__ == "__main__":
    # 範例：畫相似度熱力圖
    plot_keyword_similarity_heatmap()

    # 範例：畫幾個情境的代表歌手文字雲
    for ctx in ["工作/讀書", "失戀/分手", "告白/求婚", "運動", "派對/狂歡", "通勤/開車"]:
        plot_context_wordcloud(ctx)
