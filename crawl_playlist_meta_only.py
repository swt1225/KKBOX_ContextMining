# crawl_playlist_meta_only.py
# -*- coding: utf-8 -*-
"""
只抓 playlist metadata（title/description），不重爬 kkbox_raw_data.csv。

需求：
- 已有 data/kkbox_raw_data.csv（需包含 playlist_id 欄）
- 專案內已有 kkbox_client.py，並提供 get_kkbox_api()（與 crawl_kkbox.py 相同用法）

輸出：
- data/kkbox_playlist_meta.csv
  欄位：playlist_id, title, description, image_url, status, error

特性：
- 支援 resume：若 out 檔已存在，會跳過已抓到的 playlist_id
- 每 save_every 筆自動落盤，避免中途斷線重來
"""

import argparse
import os
import time
import pandas as pd

from kkbox_client import get_kkbox_api  # 與 crawl_kkbox.py 相同來源


REQUIRED_COLS = ["playlist_id"]


def _read_unique_playlist_ids(raw_csv: str):
    df = pd.read_csv(raw_csv)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"raw_data 缺少欄位: {missing}")

    ids = (
        df["playlist_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    # 去掉空字串
    ids = [x for x in ids if x]
    return ids


def _load_existing_meta(out_csv: str):
    if not os.path.exists(out_csv):
        return {}, pd.DataFrame(columns=["playlist_id", "title", "description", "image_url", "status", "error"])

    m = pd.read_csv(out_csv)
    if "playlist_id" not in m.columns:
        return {}, pd.DataFrame(columns=["playlist_id", "title", "description", "image_url", "status", "error"])

    m["playlist_id"] = m["playlist_id"].astype(str).str.strip()
    existing = {pid for pid in m["playlist_id"].dropna().astype(str).str.strip().tolist() if pid}
    return existing, m


def _fetch_playlist_meta(kkbox, playlist_id: str, terr: str):
    """
    呼叫 KKBOX shared_playlist_fetcher.fetch_shared_playlist
    回傳：title, description, image_url, status, error
    """
    try:
        data = kkbox.shared_playlist_fetcher.fetch_shared_playlist(playlist_id, terr=terr)
        title = (data.get("title") or "").strip()
        desc = (data.get("description") or "").strip()

        # 取一張圖（若有）
        image_url = ""
        imgs = data.get("images") or []
        for im in imgs:
            u = (im.get("url") or "").strip()
            if u:
                image_url = u
                break

        return title, desc, image_url, "ok", ""
    except Exception as e:
        # 失敗也要記錄，避免反覆重抓
        return "", "", "", "error", str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/kkbox_raw_data.csv", help="輸入 raw_data CSV（含 playlist_id）")
    ap.add_argument("--out", default="data/kkbox_playlist_meta.csv", help="輸出 meta CSV")
    ap.add_argument("--terr", default="TW", help="territory，例如 TW")
    ap.add_argument("--sleep", type=float, default=0.35, help="每次 API 呼叫後 sleep 秒數（避免過快）")
    ap.add_argument("--save-every", type=int, default=50, help="每抓到幾筆就存檔一次")
    ap.add_argument("--limit", type=int, default=0, help="只抓前 N 筆（0 表示不限制，用於測試）")
    args = ap.parse_args()

    playlist_ids = _read_unique_playlist_ids(args.raw)
    if args.limit and args.limit > 0:
        playlist_ids = playlist_ids[: args.limit]

    existing_set, meta_df = _load_existing_meta(args.out)

    # 需要抓的清單
    todo = [pid for pid in playlist_ids if pid not in existing_set]

    print(f"[Meta] raw unique playlist_id = {len(playlist_ids)}")
    print(f"[Meta] already in out = {len(existing_set)}")
    print(f"[Meta] to fetch = {len(todo)}")
    if not todo:
        print("[Meta] nothing to do.")
        return

    kkbox = get_kkbox_api()

    new_rows = []
    done = 0
    for i, pid in enumerate(todo, 1):
        title, desc, image_url, status, err = _fetch_playlist_meta(kkbox, pid, args.terr)
        new_rows.append({
            "playlist_id": pid,
            "title": title,
            "description": desc,
            "image_url": image_url,
            "status": status,
            "error": err,
        })

        done += 1

        # 進度列印
        if done == 1 or done % 10 == 0:
            print(f"[Meta] {done}/{len(todo)} fetched... (last={pid}, status={status})")

        # 定期落盤
        if done % args.save_every == 0:
            meta_df = pd.concat([meta_df, pd.DataFrame(new_rows)], ignore_index=True)
            # 去重：以 playlist_id 為主，保留先出現的（通常是 ok）
            meta_df["playlist_id"] = meta_df["playlist_id"].astype(str).str.strip()
            meta_df = meta_df.drop_duplicates(subset=["playlist_id"], keep="first")

            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            meta_df.to_csv(args.out, index=False, encoding="utf-8-sig")
            new_rows = []
            print(f"[Meta] saved -> {args.out}")

        time.sleep(max(0.0, args.sleep))

    # 收尾落盤
    if new_rows:
        meta_df = pd.concat([meta_df, pd.DataFrame(new_rows)], ignore_index=True)
        meta_df["playlist_id"] = meta_df["playlist_id"].astype(str).str.strip()
        meta_df = meta_df.drop_duplicates(subset=["playlist_id"], keep="first")

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        meta_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[Meta] done. output -> {args.out}")


if __name__ == "__main__":
    main()
