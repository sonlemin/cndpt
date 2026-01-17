# src/02_scrape_detail_topcv.py
import pandas as pd
import requests

from config import RAW_LIST_PATH, RAW_DETAIL_PATH, HEADERS
from utils import fetch_html, html_to_text, clean_url, sleep_random

def scrape_detail():
    df = pd.read_csv(RAW_LIST_PATH)
    df["link"] = df["link"].map(clean_url)

    # resume nếu chạy lại
    try:
        done = pd.read_csv(RAW_DETAIL_PATH)
        rows = done.to_dict("records")
        done_links = set(done["link"].astype(str))
    except:
        rows, done_links = [], set()

    s = requests.Session()

    for i, (title, link) in enumerate(zip(df["tieu_de"], df["link"]), start=1):
        if link in done_links:
            continue

        print(f"[{len(done_links)+1}/{len(df)}] {title}")
        html = fetch_html(s, link, headers=HEADERS)
        content = html_to_text(html) if html else ""

        rows.append({
            "tieu_de": title,
            "link": link,
            "noi_dung": content
        })
        done_links.add(link)

        # lưu checkpoint mỗi 10 tin
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(RAW_DETAIL_PATH, index=False, encoding="utf-8-sig")

        sleep_random(2.0, 5.0)

    pd.DataFrame(rows).to_csv(RAW_DETAIL_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Xong detail: {RAW_DETAIL_PATH} | Rows={len(rows)}")

if __name__ == "__main__":
    scrape_detail()
