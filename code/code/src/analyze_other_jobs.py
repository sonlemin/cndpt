#!/usr/bin/env python3
"""
analyze_other_jobs.py

Script phân tích các job titles trong nhóm "other" để tìm patterns
và đề xuất cách cải thiện classification.

USAGE:
    cd ~/workspace/github.com/sonlemin/cndpt/code/code
    python src/analyze_other_jobs.py
"""

import pandas as pd
from collections import Counter
import re

def extract_keywords(title):
    """Extract potential keywords from job title"""
    # Lowercase and split
    words = title.lower().split()
    
    # Common keywords to look for
    keywords = []
    
    # Check for specific patterns
    patterns = {
        'senior': r'\bsenior\b|\bsr\b',
        'junior': r'\bjunior\b|\bjr\b',
        'lead': r'\blead\b|\bleader\b',
        'manager': r'\bmanager\b|\bquản lý\b',
        'developer': r'\bdeveloper\b|\bdev\b|\blập trình\b',
        'engineer': r'\bengineer\b|\bkỹ sư\b',
        'specialist': r'\bspecialist\b|\bchuyên viên\b',
        'consultant': r'\bconsultant\b|\btư vấn\b',
        'architect': r'\barchitect\b|\bkiến trúc\b',
        'admin': r'\badmin\b|\bquản trị\b',
        'support': r'\bsupport\b|\bhỗ trợ\b',
        'coordinator': r'\bcoordinator\b|\bđiều phối\b',
        'analyst': r'\banalyst\b',
        'tester': r'\btester\b|\btest\b',
        'designer': r'\bdesigner\b|\bthiết kế\b',
    }
    
    title_lower = title.lower()
    for keyword, pattern in patterns.items():
        if re.search(pattern, title_lower):
            keywords.append(keyword)
    
    return keywords

def analyze_other_jobs(csv_path):
    """Analyze job titles in 'other' group"""
    print("="*80)
    print("🔍 PHÂN TÍCH NHÓM 'OTHER'")
    print("="*80)
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        print("   Hãy chạy 04_extract_features_improved.py trước")
        return
    
    # Filter "other" group
    other_df = df[df["job_group"] == "other"].copy()
    total_other = len(other_df)
    total_jobs = len(df)
    
    print(f"\n📊 Tổng quan:")
    print(f"  Total jobs: {total_jobs}")
    print(f"  'other' jobs: {total_other} ({total_other/total_jobs*100:.1f}%)")
    print(f"  Cần phân loại: {total_other} jobs")
    
    # Show sample titles
    print(f"\n📄 Mẫu 30 job titles trong 'other' (ngẫu nhiên):")
    print("-" * 80)
    sample = other_df["tieu_de"].sample(min(30, len(other_df))).tolist()
    for i, title in enumerate(sample, 1):
        print(f"{i:2d}. {title}")
    
    # Analyze keywords
    print(f"\n🔤 Phân tích từ khóa phổ biến:")
    print("-" * 80)
    
    all_keywords = []
    for title in other_df["tieu_de_clean"]:
        keywords = extract_keywords(title)
        all_keywords.extend(keywords)
    
    keyword_counts = Counter(all_keywords)
    for keyword, count in keyword_counts.most_common(20):
        pct = count / total_other * 100
        print(f"  {keyword:20s}: {count:4d} jobs ({pct:5.1f}%)")
    
    # Analyze common words
    print(f"\n💬 Từ xuất hiện nhiều trong job titles:")
    print("-" * 80)
    
    all_words = []
    for title in other_df["tieu_de_clean"]:
        words = title.split()
        all_words.extend([w for w in words if len(w) > 3])  # Skip short words
    
    word_counts = Counter(all_words)
    for word, count in word_counts.most_common(30):
        pct = count / total_other * 100
        print(f"  {word:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Suggest patterns
    print(f"\n💡 ĐỀ XUẤT PATTERNS MỚI:")
    print("-" * 80)
    print("\nDựa trên phân tích, có thể thêm các patterns sau vào config:")
    print("")
    
    suggestions = []
    
    # Check for common patterns
    if 'developer' in keyword_counts and keyword_counts['developer'] > 10:
        suggestions.append(("software_engineer", r"\bdeveloper\b|\bdev\b|\blập trình viên\b"))
    
    if 'engineer' in keyword_counts and keyword_counts['engineer'] > 10:
        suggestions.append(("software_engineer", r"\bengineer\b|\bkỹ sư\b"))
    
    if 'senior' in keyword_counts and keyword_counts['senior'] > 10:
        print("⚠️  Nhiều 'senior' titles - Có thể cần xử lý level riêng")
    
    if 'support' in keyword_counts and keyword_counts['support'] > 5:
        suggestions.append(("support", r"\bsupport\b|\bhỗ trợ\b|\bhelp desk\b"))
    
    if 'admin' in keyword_counts and keyword_counts['admin'] > 5:
        suggestions.append(("admin", r"\badmin\b|\bquản trị\b|\bsystem admin\b"))
    
    # Print suggestions
    for group, pattern in suggestions:
        print(f"  ('{group}', r'{pattern}'),")
    
    # Export full list for manual review
    output_file = "data/processed/other_jobs_analysis.csv"
    other_df[["tieu_de", "tieu_de_clean"]].to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n💾 Đã xuất full list ra: {output_file}")
    print("   Review file này để tìm thêm patterns!")
    
    print("\n" + "="*80)
    print("✅ HOÀN TẤT PHÂN TÍCH")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review suggestions phía trên")
    print("  2. Check file: data/processed/other_jobs_analysis.csv")
    print("  3. Update config với patterns mới")
    print("  4. Re-run: python src/04_extract_features_improved.py")

if __name__ == "__main__":
    analyze_other_jobs("data/processed/topcv_it_features.csv")