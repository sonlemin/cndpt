"""
04_extract_features_improved.py

Trích xuất features từ dữ liệu đã clean - VERSION CẢI THIỆN

IMPROVEMENTS:
1. ✅ Better job group classification (reduce "other" from 65% to ~20%)
2. ✅ Enhanced salary extraction (support USD, better patterns)
3. ✅ Add experience years extraction
4. ✅ Better logging and statistics

CHANGES:
- Use config_improved.py (13 job groups instead of 7)
- Better regex patterns
- More detailed logging

USAGE:
    python src/04_extract_features_improved.py
"""

import re
import pandas as pd
from pathlib import Path

# Import từ config improved
try:
    from config_improved import (
        CLEAN_PATH,
        FEATURES_PATH,
        SKILL_PATTERNS,
        JOB_GROUP_RULES,
        DEFAULT_JOB_GROUP,
    )
    print("✅ Using config_improved.py")
except ImportError:
    print("⚠️  Không tìm thấy config_improved.py, using config.py")
    from config import (
        CLEAN_PATH,
        FEATURES_PATH,
        SKILL_PATTERNS,
        JOB_GROUP_RULES,
        DEFAULT_JOB_GROUP,
    )

# ============================================================
# IMPROVED FUNCTIONS
# ============================================================

def detect_job_group(title_clean: str) -> str:
    """
    Phát hiện job group từ title - VERSION CẢI THIỆN
    
    IMPROVEMENTS:
    - Loop qua rules theo thứ tự (specific → general)
    - Stop at first match
    - Better patterns in config_improved.py
    
    Args:
        title_clean: Cleaned job title (lowercase)
    
    Returns:
        Job group name or "other"
    """
    if not isinstance(title_clean, str) or not title_clean:
        return DEFAULT_JOB_GROUP
    
    # Loop theo thứ tự (specific first)
    for group, pattern in JOB_GROUP_RULES:
        if re.search(pattern, title_clean, re.IGNORECASE):
            return group
    
    return DEFAULT_JOB_GROUP


def extract_salary(text: str):
    """
    Trích xuất lương - VERSION CẢI THIỆN
    
    IMPROVEMENTS:
    - Hỗ trợ USD (convert to VND triệu)
    - Patterns linh hoạt hơn
    - Handle edge cases
    
    Patterns:
    1. "15-25 triệu" → (15, 25, 20)
    2. "20 triệu" → (20, 20, 20)
    3. "1000-1500 USD" → (23, 34.5, 28.75) - convert using 1 USD = 23k VND
    4. "thỏa thuận" / "negotiable" → (None, None, None)
    
    Returns:
        Tuple of (min_million, max_million, avg_million)
    """
    if not isinstance(text, str) or pd.isna(text):
        return (None, None, None)
    
    text = text.lower()
    
    # Check thỏa thuận / negotiable
    if re.search(r'thỏa thuận|thoả thuận|協議|negotiable|competitive|liên hệ', text):
        return (None, None, None)
    
    # Pattern 1: USD range "1000-1500 USD"
    usd_range = r'(\d+(?:[,\.]\d+)?)\s*[-~tới đến]+\s*(\d+(?:[,\.]\d+)?)\s*(?:usd|\$)'
    match = re.search(usd_range, text)
    if match:
        min_usd = float(match.group(1).replace(',', ''))
        max_usd = float(match.group(2).replace(',', ''))
        # 1 USD ≈ 23,000 VND = 0.023 triệu VND
        min_vnd = min_usd * 0.023
        max_vnd = max_usd * 0.023
        avg_vnd = (min_vnd + max_vnd) / 2
        return (min_vnd, max_vnd, avg_vnd)
    
    # Pattern 2: Single USD "1500 USD"
    usd_single = r'(\d+(?:[,\.]\d+)?)\s*(?:usd|\$)'
    match = re.search(usd_single, text)
    if match:
        usd = float(match.group(1).replace(',', ''))
        vnd = usd * 0.023
        return (vnd, vnd, vnd)
    
    # Pattern 3: VND range "15-25 triệu"
    # Support: -, ~, tới, đến as separators
    vnd_range = r'(\d+(?:[,\.]\d+)?)\s*[-~tới đến]+\s*(\d+(?:[,\.]\d+)?)\s*(?:triệu|tr|trieu|million|triệu đồng)'
    match = re.search(vnd_range, text)
    if match:
        min_sal = float(match.group(1).replace(',', '.'))
        max_sal = float(match.group(2).replace(',', '.'))
        avg_sal = (min_sal + max_sal) / 2
        return (min_sal, max_sal, avg_sal)
    
    # Pattern 4: Single VND "20 triệu"
    vnd_single = r'(\d+(?:[,\.]\d+)?)\s*(?:triệu|tr|trieu|million|triệu đồng)\b'
    match = re.search(vnd_single, text)
    if match:
        sal = float(match.group(1).replace(',', '.'))
        return (sal, sal, sal)
    
    return (None, None, None)


def extract_skills(text: str):
    """
    Trích xuất skills từ text - GIỮ NGUYÊN
    
    Logic: Loop qua SKILL_PATTERNS
    """
    if not isinstance(text, str) or pd.isna(text):
        return []
    
    found = []
    text_lower = text.lower()
    
    for skill, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, text_lower):
            found.append(skill)
    
    return found


def extract_experience_years(text: str):
    """
    Trích xuất số năm kinh nghiệm - THÊM MỚI
    
    Patterns:
    1. "3 năm kinh nghiệm" → 3.0
    2. "2-3 năm" → 2.5 (average)
    3. "fresher" / "không yêu cầu" → 0.0
    4. "5+ years" → 5.0
    
    Returns:
        Float (years) or None
    """
    if not isinstance(text, str) or pd.isna(text):
        return None
    
    text = text.lower()
    
    # Check fresher / no experience
    if re.search(r'fresher|không yêu cầu kinh nghiệm|no experience|entry level', text):
        return 0.0
    
    # Pattern 1: Range "2-3 năm" → average
    pattern_range = r'(\d+)\s*[-~tới đến]+\s*(\d+)\s*(?:năm|years?|yr)'
    match = re.search(pattern_range, text)
    if match:
        min_exp = float(match.group(1))
        max_exp = float(match.group(2))
        return (min_exp + max_exp) / 2
    
    # Pattern 2: "5+ năm"
    pattern_plus = r'(\d+)\+\s*(?:năm|years?|yr)'
    match = re.search(pattern_plus, text)
    if match:
        return float(match.group(1))
    
    # Pattern 3: "3 năm"
    pattern_single = r'(\d+)\s*(?:năm|years?|yr)'
    matches = re.findall(pattern_single, text)
    if matches:
        # Take first occurrence
        return float(matches[0])
    
    return None


# ============================================================
# MAIN
# ============================================================

def print_statistics(df):
    """Print detailed statistics"""
    print("\n" + "="*80)
    print("📊 STATISTICS")
    print("="*80)
    
    # Job groups
    print("\n🏷️  Job Groups Distribution:")
    group_counts = df["job_group"].value_counts()
    total = len(df)
    for group, count in group_counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {group:20s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Salary
    has_salary = df["has_salary"].sum()
    no_salary = total - has_salary
    print(f"\n💰 Salary Info:")
    print(f"  Có lương:       {has_salary:4d} ({has_salary/total*100:5.1f}%)")
    print(f"  Không có lương: {no_salary:4d} ({no_salary/total*100:5.1f}%)")
    
    if has_salary > 0:
        salary_df = df[df["salary_avg"].notna()]
        print(f"  Lương trung bình: {salary_df['salary_avg'].mean():.1f} triệu")
        print(f"  Lương min:        {salary_df['salary_avg'].min():.1f} triệu")
        print(f"  Lương max:        {salary_df['salary_avg'].max():.1f} triệu")
    
    # Experience
    has_exp = df["experience_years"].notna().sum()
    if has_exp > 0:
        exp_df = df[df["experience_years"].notna()]
        print(f"\n👔 Experience Requirements:")
        print(f"  Có yêu cầu kinh nghiệm: {has_exp:4d} ({has_exp/total*100:5.1f}%)")
        print(f"  Kinh nghiệm TB: {exp_df['experience_years'].mean():.1f} năm")
    
    # Skills
    print(f"\n🔧 Skills:")
    print(f"  Trung bình skills/job: {df['n_skills'].mean():.1f}")
    print(f"  Max skills in a job:   {df['n_skills'].max()}")
    print(f"  Jobs with 0 skills:    {(df['n_skills'] == 0).sum()}")
    
    print("="*80)


def main():
    """Main extraction function"""
    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU TRÍCH XUẤT FEATURES (IMPROVED VERSION)")
    print("="*80)
    
    # Load data
    print(f"\n📂 Loading data from: {CLEAN_PATH}")
    try:
        df = pd.read_csv(CLEAN_PATH)
        print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"❌ File not found: {CLEAN_PATH}")
        print("   Please run 03_preprocess_clean.py first")
        return
    
    # 1. Job groups
    print("\n" + "-"*80)
    print("📊 Extracting job groups...")
    df["job_group"] = df["tieu_de_clean"].apply(detect_job_group)
    print(f"✅ Done. Groups: {df['job_group'].nunique()}")
    
    # 2. Salary
    print("\n" + "-"*80)
    print("💰 Extracting salary...")
    salary = df["noi_dung_clean"].apply(extract_salary)
    df["salary_min"] = salary.apply(lambda x: x[0])
    df["salary_max"] = salary.apply(lambda x: x[1])
    df["salary_avg"] = salary.apply(lambda x: x[2])
    df["has_salary"] = df["salary_avg"].notna().astype(int)
    has_salary = df["has_salary"].sum()
    print(f"✅ Done. Found salary in {has_salary}/{len(df)} jobs ({has_salary/len(df)*100:.1f}%)")
    
    # 3. Skills
    print("\n" + "-"*80)
    print("🔧 Extracting skills...")
    df["skills"] = df["noi_dung_clean"].apply(extract_skills)
    df["n_skills"] = df["skills"].apply(len)
    df["skills_str"] = df["skills"].apply(lambda lst: ",".join(lst))
    print(f"✅ Done. Average {df['n_skills'].mean():.1f} skills/job")
    
    # 4. Experience (NEW!)
    print("\n" + "-"*80)
    print("👔 Extracting experience years...")
    df["experience_years"] = df["noi_dung_clean"].apply(extract_experience_years)
    has_exp = df["experience_years"].notna().sum()
    print(f"✅ Done. Found experience in {has_exp}/{len(df)} jobs ({has_exp/len(df)*100:.1f}%)")
    
    # Save
    print("\n" + "-"*80)
    print(f"💾 Saving to: {FEATURES_PATH}")
    df.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {df.shape}")
    
    # Statistics
    print_statistics(df)
    
    # Sample
    print("\n" + "="*80)
    print("📄 SAMPLE (first 5 rows)")
    print("="*80)
    sample_df = df[["tieu_de", "job_group", "salary_avg", "experience_years", "n_skills"]].head(5)
    print(sample_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ HOÀN TẤT!")
    print("="*80)
    print(f"\n📊 Next steps:")
    print(f"  1. Run: python src/05_eda_visualize.py")
    print(f"  2. Check: {FEATURES_PATH}")
    print(f"  3. View charts in: reports/figures/")


if __name__ == "__main__":
    main()