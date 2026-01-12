# artist_keywords.py
# -*- coding: utf-8 -*-
import re
from collections import Counter

import jieba
from wordcloud import WordCloud
import io

from kkbox_client import get_kkbox_api
import pickle   # 如果你用 playlist_pool.pkl
# ...

# 6 個情境的關鍵字（可直接沿用你 crawl_kkbox 的版本）
CONTEXT_QUERIES = {
    "工作/讀書": [
        "工作", "上班", "上班族",
        "讀書", "念書", "讀書用", "讀書歌單", "讀書音樂",
        "專注", "專心", "集中精神",
        "study", "study music", "focus", "deep focus",
        "work", "work music"
    ],
    "失戀/分手": [
        "失戀", "失戀歌單", "失戀情歌", "失戀歌曲",
        "分手", "分手快樂", "分手歌",
        "心碎", "難過", "難過歌",
        "療傷", "療傷歌",
        "哭", "想哭", "傷心",
        "sad", "sad songs", "sad playlist",
        "heartbreak", "heartbroken", "breakup", "broken heart", "crying"
    ],
    "告白/求婚": [
        "告白/求婚", "表白", "情書", "示愛",
        "告白情歌", "告白專用", "浪漫",
        "暗戀", "暗戀情歌",
        "戀愛", "戀愛告白", "愛情告白",
        "求婚", "婚禮", "婚禮進場",
        "love", "love songs", "love song",
        "confession", "proposal", "wedding proposal", "romantic"
    ],
    "運動": [
        "運動", "健身房", "健身歌單",
        "運動", "運動歌單", "運動嗨歌",
        "重訓", "訓練", "練腿", "爆汗", "燃脂", "增肌",
        "跑步", "慢跑", "跑步歌單", "有氧", "有氧運動",
        "gym", "gym music", "workout", "workout mix",
        "training", "fitness", "cardio",
        "running", "jogging", "squat", "deadlift",
        "HIIT", "crossfit", "pump", "gym motivation"
    ],
"派對/狂歡": [
        "派對/狂歡", "派對歌單", "派對嗨歌",
        "嗨", "嗨歌", "熱鬧", "嗨爆",
        "夜店", "舞曲", "舞池",
        "狂歡", "電音", "EDM", "酒吧",
        "跨年", "慶功", "蹦迪",
        "party", "party songs", "party mix",
        "club", "clubbing", "dance party",
        "festival", "rave", "night out", "drinking"
    ],
    "通勤/開車": [
        "通勤/開車", "通勤歌單", "上班路上", "上下班",
        "開車", "開車歌單", "開車音樂",
        "搭車", "坐車", "公車", "捷運",
        "兜風", "公路", "公路旅行", "自駕",
        "drive", "driving", "driving music",
        "commute", "commuting", "commute playlist",
        "road trip", "roadtrip", "highway",
        "car ride", "car music",
    ],
}


STOPWORDS = set([
    "歌單", "精選", "熱門", "人氣", "歌曲", "音樂", "playlist",
    "official", "official playlist", "best", "hits", "collection",
    "華語","中文","100","曲目","完整","步頻","更新",
    "必備","必聽","西洋","經典","K-POP"
    # 常見無意義字
    "的", "和", "與", "在", "為", "用", "版",
    "這個","這樣","這些","就是","還是","那些","必"
])
# app.py

CONTEXT_ORDER = ['工作/讀書', '失戀/分手', '告白/求婚', '運動', '派對/狂歡', '通勤/開車']


def get_artist_context_stats(artist_name: str,
                             max_playlists_per_artist: int = 200):
    """
    從 kkbox_raw_data.csv 中取出：
    - 此歌手在哪些歌單出現
    - 每個情境中，含此歌手的歌單數量（去重後）

    回傳：
        playlist_ids: list[str]   # 後面拿去抓 title
        context_counts: dict[context, count]
    """
    # 只看名字完全相等的歌手（簡單版；之後要支援 id 再延伸）
    df_a = df_raw[df_raw["artist"] == artist_name].copy()
    if df_a.empty:
        return [], {ctx: 0 for ctx in CONTEXT_ORDER}

    # 每個情境中，統計「不同歌單」數量
    # groupby 後 nunique 保證同一歌單只算一次
    g = df_a.groupby("context")["playlist_id"].nunique()
    context_counts = {ctx: int(g.get(ctx, 0)) for ctx in CONTEXT_ORDER}

    # 給文字雲用的歌單 id（限制最多 N 張，避免 API 過多）
    playlist_ids = df_a["playlist_id"].dropna().astype(str).unique().tolist()
    if len(playlist_ids) > max_playlists_per_artist:
        playlist_ids = playlist_ids[:max_playlists_per_artist]

    return playlist_ids, context_counts
playlist_title_cache = {}  # playlist_id -> (title, description)

def get_playlist_title_desc(kkbox, playlist_id: str):
    """
    從 KKBOX API 抓歌單的 title / description，並快取結果。
    """
    if playlist_id in playlist_title_cache:
        return playlist_title_cache[playlist_id]

    try:
        data = kkbox.shared_playlist_fetcher.fetch_shared_playlist(
            playlist_id,
            terr="TW"
        )
    except Exception as e:
        print(f"[WARN] fetch_shared_playlist({playlist_id}) error:", e)
        playlist_title_cache[playlist_id] = ("", "")
        return "", ""

    title = data.get("title", "") or ""
    desc = data.get("description", "") or ""
    playlist_title_cache[playlist_id] = (title, desc)
    return title, desc
def detect_contexts_for_playlist(title: str, desc: str):
    """
    根據歌單 title + description 判斷屬於哪些情境。
    只要出現該情境任一關鍵字就算命中。

    回傳：list，例如 ['失戀', '']
    """
    text = (title or "") + " " + (desc or "")
    text_lower = text.lower()

    matched = []

    for ctx, keywords in CONTEXT_QUERIES.items():
        for kw in keywords:
            if re.search(r'[\u4e00-\u9fff]', kw):
                # 中文關鍵字：大小寫沒差
                if kw in text:
                    matched.append(ctx)
                    break
            else:
                # 英文關鍵字：用 lower 比對
                if kw.lower() in text_lower:
                    matched.append(ctx)
                    break

    return matched
def fetch_playlists_for_artist(artist_name, max_playlists=50):
    """
    用『歌手名字』搜尋 KKBOX 歌單：
    - search_fetcher.search(keyword, types=['playlist'], terr='TW')
    - 回傳 [(title, description), ...]
    """
    kkbox = get_kkbox_api()

    search_results = kkbox.search_fetcher.search(
        artist_name,
        types=['playlist'],
        terr='TW'
    )

    playlists_all = search_results.get("playlists", {}).get("data", [])
    playlists = playlists_all[:max_playlists]

    results = []
    for pl in playlists:
        title = pl.get("title", "") or ""
        desc = pl.get("description", "") or ""
        results.append((title, desc))

    return results


def extract_keywords_from_titles(titles, artist_name):
    """
    給一堆歌單標題/描述，抽出關鍵字頻率：
    - 中文用 jieba 斷詞
    - 英文用簡單 split
    - 移除停用詞 / 歌手名
    回傳 Counter
    """
    full_text = " ".join(titles)

    # 先處理英文：全部轉小寫
    full_text_lower = full_text.lower()

    # jieba 斷詞（會同時切中英）
    words = jieba.lcut(full_text)

    # 歌手名本身也視為停用詞
    artist_name_clean = artist_name.strip()
    # 英文版歌手名（全部小寫）
    artist_name_lower = artist_name_clean.lower()

    stopwords = set(STOPWORDS)
    stopwords.add(artist_name_clean)
    stopwords.add(artist_name_lower)

    # 把括號內容去掉，用來過濾像「周杰倫 (Jay Chou)」這種重複資訊
    def remove_parentheses(text):
        return re.sub(r"[\(\（][^)\）]*[\)\）]", "", text)

    filtered = []
    for w in words:
        w = w.strip()
        if not w:
            continue

        # 英文統一小寫
        w_lower = w.lower()

        # 移除括號內容
        w_clean = remove_parentheses(w)
        if not w_clean:
            continue

        # 長度 1 的字大多沒資訊（例如「的」、「之」）
        if len(w_clean) == 1:
            continue

        # 停用詞過濾
        if w_clean in stopwords or w_lower in stopwords:
            continue

        filtered.append(w_clean)

    counter = Counter(filtered)
    return counter


def generate_wordcloud_image(freq_counter, width=800, height=400,
                             font_path="C:/Windows/Fonts/msjh.ttc"):
    """
    根據 Counter 產生文字雲 PNG（回傳 bytes）
    """
    if not freq_counter:
        return None

    wc = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        background_color="white",
        prefer_horizontal=1.0
    )

    wc.generate_from_frequencies(freq_counter)

    img_bytes = io.BytesIO()
    wc.to_image().save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes
