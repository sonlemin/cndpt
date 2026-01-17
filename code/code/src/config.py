# src/config.py
BASE_URL = "https://www.topcv.vn"
START_URL = "https://www.topcv.vn/tim-viec-lam-it"

RAW_LIST_PATH   = "data/raw/topcv_it_list.csv"
RAW_DETAIL_PATH = "data/raw/topcv_it_detail.csv"

CLEAN_PATH      = "data/processed/topcv_it_clean.csv"
FEATURES_PATH   = "data/processed/topcv_it_features.csv"

FIG_DIR         = "reports/figures"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9"
}
