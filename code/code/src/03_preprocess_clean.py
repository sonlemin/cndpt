# src/03_preprocess_clean.py
import pandas as pd
from config import RAW_DETAIL_PATH, CLEAN_PATH
from utils import clean_text_vn, clean_url

def preprocess():
    df = pd.read_csv(RAW_DETAIL_PATH)

    df["link"] = df["link"].astype(str).map(clean_url)

    # bỏ trùng và thiếu
    df = df.drop_duplicates(subset=["link"])
    df = df.dropna(subset=["tieu_de", "link", "noi_dung"])

    # clean
    df["tieu_de_clean"] = df["tieu_de"].astype(str).map(clean_text_vn)
    df["noi_dung_clean"] = df["noi_dung"].astype(str).map(clean_text_vn)

    # bỏ các dòng content quá ngắn (lọc rác)
    df = df[df["noi_dung_clean"].str.len() >= 200].copy()

    df.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
    print("✅ Saved:", CLEAN_PATH, "| Shape:", df.shape)

if __name__ == "__main__":
    preprocess()
