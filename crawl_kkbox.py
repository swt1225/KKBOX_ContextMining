# crawl_kkbox.py
import pandas as pd
import time
from kkbox_client import get_kkbox_api


def crawl_kkbox_data(context_queries: dict,
                     playlists_per_query: int = 20,
                     sleep_sec: float = 0.5) -> pd.DataFrame:
    """
    context_queries: dict
        key   = 情境名稱（例如 '失戀/分手', '告白/求婚', '運動'）
        value = 該情境底下的多個搜尋關鍵字 list，
                例如 ['失戀', '分手', '心碎']

    回傳欄位：
        context      : 情境標籤（失戀 / 告白 / 健身…）
        search_term  : 實際拿去 search 的關鍵字
        playlist_id  : 歌單 ID
        artist       : 歌手名稱
    """
    kkbox = get_kkbox_api()
    raw_data = []

    for context, query_list in context_queries.items():
        print(f"[Crawl] 情境：{context}，關鍵字組合：{query_list}")

        # 避免：同一情境下，同一歌單被不同關鍵字抓到多次 → 只取一次
        seen_playlists_in_context = set()

        for query in query_list:
            print(f"  - 搜尋關鍵字：{query}")
            try:
                search_results = kkbox.search_fetcher.search(
                    query,               # 關鍵字（位置參數）
                    types=['playlist'],  # 找歌單
                    terr='TW'
                )
            except Exception as e:
                print(f"    搜尋 {query} 發生錯誤：{e}")
                continue

            playlists_all = search_results.get('playlists', {}).get('data', [])
            playlists = playlists_all[:playlists_per_query]
            playlist_meta = {}  # playlist_id -> {"playlist_id":..., "title":..., "description":...}
            for pl in playlists:
                pl_id = pl['id']

                # 同一情境 + 同一歌單，只處理一次
                if pl_id in seen_playlists_in_context:
                    continue
                seen_playlists_in_context.add(pl_id)
                if pl_id not in playlist_meta:
                    try:
                        info = kkbox.shared_playlist_fetcher.fetch_shared_playlist(pl_id, terr="TW")
                        playlist_meta[pl_id] = {
                            "playlist_id": pl_id,
                            "title": (info.get("title") or "").strip(),
                            "description": (info.get("description") or "").strip(),
                        }
                    except Exception:
                        playlist_meta[pl_id] = {"playlist_id": pl_id, "title": "", "description": ""}
                try:
                    tracks_data = kkbox.shared_playlist_fetcher.fetch_tracks_of_shared_playlists(
                        pl_id,
                        terr='TW'
                    )
                    tracks = tracks_data['data']
                except Exception as e:
                    print(f"    讀取歌單 {pl_id} 失敗：{e}")
                    continue

                for track in tracks:
                    # 有些 track 可能沒有 album / artist，先保護一下
                    album = track.get('album', {})
                    artist_info = album.get('artist')
                    if not artist_info:
                        continue

                    artist_name = artist_info.get('name')
                    if not artist_name:
                        continue

                    raw_data.append({
                        "context": context,        # 情境標籤
                        "search_term": query,      # 實際搜尋字
                        "playlist_id": pl_id,
                        "artist": artist_name,
                    })

                time.sleep(sleep_sec)  # 避免 API 過度頻繁

    df = pd.DataFrame(raw_data)
    meta_df = pd.DataFrame(list(playlist_meta.values()))
    return df, meta_df


if __name__ == "__main__":
    # 範例：一個情境對多個關鍵字
    context_queries = {
    "工作/讀書": [
        # 中文
        "工作", "上班", "上班族",
        "讀書", "念書", "讀書用", "讀書歌單", "讀書音樂",
        "專注", "專心", "集中精神","提神"
        # 英文 / 混用
        "study", "study music", "focus", "deep focus",
        "work", "work music"
    ],
    "失戀/分手": [
        # 中文
        "失戀", "失戀歌單", "失戀情歌", "失戀歌曲",
        "分手", "分手快樂", "分手歌",
        "心碎", "難過", "難過歌",
        "療傷", "療傷歌",
        "哭", "想哭", "傷心",
        # 英文
        "sad", "sad songs", "sad playlist",
        "heartbreak", "heartbroken", "breakup", "broken heart", "crying"
    ],
    "告白/求婚": [
        # 中文
        "告白", "表白", "情書", "示愛",
        "告白情歌", "告白專用", "浪漫告白",
        "暗戀", "暗戀情歌",
        "戀愛", "戀愛告白", "愛情告白",
        "求婚", "婚禮", "婚禮進場",
        # 英文
        "love", "love songs", "love song",
        "confession", "proposal", "wedding proposal", "romantic"
    ],
    "運動": [
        # 中文
        "健身", "健身房", "健身歌單",
        "運動", "運動歌單", "運動嗨歌",
        "重訓", "訓練", "練腿", "爆汗", "燃脂", "增肌",
        "跑步", "慢跑", "跑步歌單", "有氧", "有氧運動",
        # 英文
        "gym", "gym music", "workout", "workout mix",
        "training", "fitness", "cardio",
        "running", "jogging", "squat", "deadlift",
        "HIIT", "crossfit", "pump", "gym motivation"
    ],
    "派對/狂歡": [
        # 中文
        "派對", "派對歌單", "派對嗨歌",
        "嗨", "嗨歌", "熱鬧", "嗨爆",
        "夜店", "舞曲", "舞池",
        "狂歡", "電音", "EDM", "酒吧",
        "跨年", "慶功", "蹦迪",
        # 英文
        "party", "party songs", "party mix",
        "club", "clubbing", "dance party",
        "festival", "rave", "night out", "drinking"
    ],
    "通勤/開車": [
        # 中文
        "通勤", "通勤歌單", "上班路上", "上下班",
        "開車", "開車歌單", "開車音樂",
        "搭車", "坐車", "公車", "捷運",
        "兜風", "公路", "公路旅行", "自駕",
        # 英文
        "drive", "driving", "driving music",
        "commute", "commuting", "commute playlist",
        "road trip", "roadtrip", "highway",
        "car ride", "car music",
    ],
}

df, meta_df  = crawl_kkbox_data(context_queries,
                          playlists_per_query=50,
                          sleep_sec=0.5)

df.to_csv("data/kkbox_raw_data.csv", index=False, encoding="utf-8-sig")
meta_df.to_csv("data/kkbox_playlist_meta.csv", index=False, encoding="utf-8-sig")
print("資料收集完成，筆數：", len(df))
