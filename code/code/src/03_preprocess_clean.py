import pandas as pd
from config import RAW_DETAIL_PATH, CLEAN_PATH
from utils import clean_text_vn, clean_url

def preprocess():
    df = pd.read_csv(RAW_DETAIL_PATH)

    df = df.dropna(subset=["tieu_de", "link", "noi_dung"]).copy()

    df["link"] = df["link"].astype(str).map(clean_url)
    df["link"] = df["link"].replace({"nan": pd.NA, "": pd.NA}).dropna()

    df = df.dropna(subset=["link"])
    df = df.drop_duplicates(subset=["link"])

    df["tieu_de_clean"] = df["tieu_de"].astype(str).map(clean_text_vn)
    df["noi_dung_clean"] = df["noi_dung"].astype(str).map(clean_text_vn)

    raw_len = df["noi_dung"].astype(str).str.len()
    clean_len = df["noi_dung_clean"].astype(str).str.len()
    df = df[(raw_len >= 800) | (clean_len >= 200)].copy()

    df.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
    print("✅ Saved:", CLEAN_PATH, "| Shape:", df.shape)

if __name__ == "__main__":
    preprocess()
