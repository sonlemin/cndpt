# src/utils.py
import time, random, re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlsplit, urlunsplit

def clean_url(u: str) -> str:
    """Bỏ query/utm để tránh trùng link"""
    parts = urlsplit(str(u))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

def html_to_text(html: str) -> str:
    """Chuyển HTML -> text thô"""
    soup = BeautifulSoup(html, "lxml")
    return " ".join(soup.get_text(" ", strip=True).split())

def fetch_html(session: requests.Session, url: str, headers=None, retry=2, timeout=20) -> str:
    for _ in range(retry):
        r = session.get(url, headers=headers, timeout=timeout)
        if r.status_code == 429:
            time.sleep(random.uniform(40, 70))  # bị chặn
            continue
        r.raise_for_status()
        return r.text
    return ""

def sleep_random(a=1.0, b=2.0):
    time.sleep(random.uniform(a, b))

def normalize_spaces(s: str) -> str:
    return " ".join(str(s).strip().split())

def clean_text_vn(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)                 # bỏ link
    text = re.sub(r"[^\wÀ-ỹ\s\+\#\.\-]", " ", text)      # giữ chữ VN + ký tự kỹ thuật
    text = re.sub(r"\s+", " ", text).strip()
    return text
