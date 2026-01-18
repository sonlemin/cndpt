"""
04_extract_features.py

Trích xuất features từ dữ liệu đã clean.

BASED ON: Code cũ (logic giữ nguyên)
UPDATED: Import từ config.py
"""

import re
import pandas as pd

# Import từ config
from config import (
    CLEAN_PATH,
    FEATURES_PATH,
    SKILL_PATTERNS,
    JOB_GROUP_RULES,
)

# ============================================================
# FUNCTIONS (giữ nguyên logic code cũ)
# ============================================================

def detect_job_group(title_clean: str) -> str:
    """
    Phát hiện job group từ title.
    
    Logic code cũ: Loop qua JOB_GROUP_RULES
    """
    for group, pattern in JOB_GROUP_RULES:
        if re.search(pattern, title_clean):
            return group
    return "other"


def extract_salary(text: str):
    """
    Trích xuất lương (VND only).
    
    Logic code cũ:
    - Thỏa thuận → (None, None, None)
    - Range: 15-25 triệu → (15, 25, 20)
    - Single: 20 triệu → (20, 20, 20)
    
    Returns:
        Tuple of (min_million, max_million, avg_million)
    """
    t = text.lower()

    # Check thỏa thuận
    if "thỏa thuận" in t or "thoả thuận" in t:
        return (None, None, None)

    # Pattern 1: Range (15-25 triệu)
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*[-~đến]+\s*(\d+(?:[\.,]\d+)?)\s*(triệu|tr)", t)
    if m:
        a = float(m.group(1).replace(",", "."))
        b = float(m.group(2).replace(",", "."))
        return (a, b, (a + b) / 2)

    # Pattern 2: Single value (20 triệu)
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(triệu|tr)\b", t)
    if m:
        a = float(m.group(1).replace(",", "."))
        return (a, a, a)

    return (None, None, None)


def extract_skills(text: str):
    """
    Trích xuất skills từ text.
    
    Logic code cũ: Loop qua SKILL_PATTERNS
    """
    found = []
    for skill, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, text):
            found.append(skill)
    return found


# ============================================================
# MAIN
# ============================================================

def main():
    """Main extraction function"""
    print("🚀 Bắt đầu trích xuất features")
    
    # Load data
    df = pd.read_csv(CLEAN_PATH)
    print(f"📂 Loaded: {df.shape}")
    
    # Job groups
    print("📊 Extracting job groups...")
    df["job_group"] = df["tieu_de_clean"].apply(detect_job_group)
    
    # Salary
    print("💰 Extracting salary...")
    salary = df["noi_dung_clean"].apply(extract_salary)
    df["salary_min"] = salary.apply(lambda x: x[0])
    df["salary_max"] = salary.apply(lambda x: x[1])
    df["salary_avg"] = salary.apply(lambda x: x[2])
    df["has_salary"] = df["salary_avg"].notna().astype(int)
    
    # Skills
    print("🔧 Extracting skills...")
    df["skills"] = df["noi_dung_clean"].apply(extract_skills)
    df["n_skills"] = df["skills"].apply(len)
    df["skills_str"] = df["skills"].apply(lambda lst: ",".join(lst))
    
    # Save
    df.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")
    print()
    print("✅ Saved:", FEATURES_PATH, "| Shape:", df.shape)
    
    # Sample
    print("\n📄 Sample:")
    print(df[["tieu_de", "job_group", "skills_str", "salary_avg"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()