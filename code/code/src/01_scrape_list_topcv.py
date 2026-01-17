# src/01_scrape_list_topcv.py
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from config import BASE_URL, START_URL, RAW_LIST_PATH, HEADERS
from utils import normalize_spaces, fetch_html, sleep_random, clean_url

def parse_jobs(html: str):
    soup = BeautifulSoup(html, "lxml")
    jobs = []
    seen = set()

    for a in soup.select("a[href*='/viec-lam/']"):
        title = normalize_spaces(a.get_text())
        href = a.get("href", "")
        if len(title) < 6:
            continue

        link = clean_url(urljoin(BASE_URL, href))
        if link in seen:
            continue

        seen.add(link)
        jobs.append({"tieu_de": title, "link": link})

    return jobs

def scrape_topcv_it(max_jobs=500, max_pages=200):
    s = requests.Session()
    all_jobs = []

    for page in range(1, max_pages + 1):
        if len(all_jobs) >= max_jobs:
            break

        url = f"{START_URL}?page={page}"
        print(f"📌 Trang {page}: {url}")

        html = fetch_html(s, url, headers=HEADERS)
        jobs = parse_jobs(html)

        if not jobs:
            print("⚠️ Không thấy job -> dừng.")
            break

        all_jobs.extend(jobs)
        df_tmp = pd.DataFrame(all_jobs).drop_duplicates(subset=["link"])
        all_jobs = df_tmp.to_dict("records")

        print(f"✅ Tổng tin hiện tại: {len(all_jobs)}")
        sleep_random(1.0, 2.0)

    return all_jobs[:max_jobs]

if __name__ == "__main__":
    jobs = scrape_topcv_it(max_jobs=500, max_pages=200)
    df = pd.DataFrame(jobs)
    df.to_csv(RAW_LIST_PATH, index=False, encoding="utf-8-sig")
    print("🎉 Hoàn tất list!")
    print("✅ Lưu:", RAW_LIST_PATH, " | Shape:", df.shape)
