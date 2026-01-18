#!/usr/bin/env python3
"""
diagnose_skills.py

Script chẩn đoán tại sao PHP/Java xuất hiện trong 100% jobs

USAGE:
    cd ~/workspace/github.com/sonlemin/cndpt/code/code
    python src/diagnose_skills.py
"""

import pandas as pd
import re
from collections import Counter

def diagnose_skill_patterns(csv_path):
    """Diagnose skill extraction issues"""
    print("="*80)
    print("🔍 CHẨN ĐOÁN SKILL PATTERNS")
    print("="*80)
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        print("   Chạy 04_extract_features_improved.py trước")
        return
    
    print(f"\n📊 Tổng quan:")
    print(f"  Total jobs: {len(df)}")
    
    # Analyze PHP
    print(f"\n" + "="*80)
    print("📌 PHÂN TÍCH: PHP")
    print("="*80)
    
    php_pattern = r"\bphp\b"
    php_count = 0
    php_samples = []
    
    for idx, row in df.head(100).iterrows():  # Check first 100 jobs
        text_combined = ""
        
        if "tieu_de_clean" in df.columns and pd.notna(row["tieu_de_clean"]):
            text_combined += str(row["tieu_de_clean"]).lower() + " "
        
        if "mo_ta_clean" in df.columns and pd.notna(row["mo_ta_clean"]):
            text_combined += str(row["mo_ta_clean"]).lower() + " "
        
        if re.search(php_pattern, text_combined, re.IGNORECASE):
            php_count += 1
            
            # Find where PHP appears
            match = re.search(r'.{0,50}\bphp\b.{0,50}', text_combined, re.IGNORECASE)
            if match and len(php_samples) < 10:
                php_samples.append({
                    'job_id': idx,
                    'title': row.get('tieu_de', 'N/A'),
                    'group': row.get('job_group', 'N/A'),
                    'context': match.group(0)
                })
    
    print(f"\n📊 Trong 100 jobs đầu tiên:")
    print(f"  PHP xuất hiện: {php_count}/100 ({php_count}%)")
    
    if php_count > 80:
        print(f"  ⚠️  WARNING: PHP xuất hiện quá nhiều (>80%)!")
        print(f"  → Pattern có thể quá rộng HOẶC data có boilerplate text")
    
    print(f"\n📄 10 mẫu context có chứa 'PHP':")
    print("-"*80)
    for i, sample in enumerate(php_samples, 1):
        print(f"\n{i}. Job: {sample['title'][:60]}")
        print(f"   Group: {sample['group']}")
        print(f"   Context: ...{sample['context']}...")
    
    # Analyze Java
    print(f"\n" + "="*80)
    print("📌 PHÂN TÍCH: JAVA")
    print("="*80)
    
    java_pattern = r"\bjava\b(?!\s*script)"
    java_count = 0
    java_samples = []
    
    for idx, row in df.head(100).iterrows():
        text_combined = ""
        
        if "tieu_de_clean" in df.columns and pd.notna(row["tieu_de_clean"]):
            text_combined += str(row["tieu_de_clean"]).lower() + " "
        
        if "mo_ta_clean" in df.columns and pd.notna(row["mo_ta_clean"]):
            text_combined += str(row["mo_ta_clean"]).lower() + " "
        
        if re.search(java_pattern, text_combined, re.IGNORECASE):
            java_count += 1
            
            match = re.search(r'.{0,50}\bjava\b.{0,50}', text_combined, re.IGNORECASE)
            if match and len(java_samples) < 10:
                java_samples.append({
                    'job_id': idx,
                    'title': row.get('tieu_de', 'N/A'),
                    'group': row.get('job_group', 'N/A'),
                    'context': match.group(0)
                })
    
    print(f"\n📊 Trong 100 jobs đầu tiên:")
    print(f"  Java xuất hiện: {java_count}/100 ({java_count}%)")
    
    if java_count > 80:
        print(f"  ⚠️  WARNING: Java xuất hiện quá nhiều (>80%)!")
    
    print(f"\n📄 10 mẫu context có chứa 'Java':")
    print("-"*80)
    for i, sample in enumerate(java_samples, 1):
        print(f"\n{i}. Job: {sample['title'][:60]}")
        print(f"   Group: {sample['group']}")
        print(f"   Context: ...{sample['context']}...")
    
    # Analyze by job group
    print(f"\n" + "="*80)
    print("📊 PHÂN TÍCH THEO JOB GROUP")
    print("="*80)
    
    print(f"\nPHP distribution by job_group (first 100 jobs):")
    php_by_group = {}
    for idx, row in df.head(100).iterrows():
        group = row.get('job_group', 'unknown')
        text = ""
        if "tieu_de_clean" in df.columns:
            text += str(row.get("tieu_de_clean", "")).lower() + " "
        if "mo_ta_clean" in df.columns:
            text += str(row.get("mo_ta_clean", "")).lower() + " "
        
        if group not in php_by_group:
            php_by_group[group] = {'total': 0, 'has_php': 0}
        php_by_group[group]['total'] += 1
        
        if re.search(php_pattern, text, re.IGNORECASE):
            php_by_group[group]['has_php'] += 1
    
    for group, stats in sorted(php_by_group.items(), key=lambda x: x[1]['has_php'], reverse=True):
        pct = stats['has_php'] / stats['total'] * 100
        print(f"  {group:20s}: {stats['has_php']:2d}/{stats['total']:2d} ({pct:5.1f}%)")
    
    # Recommendations
    print(f"\n" + "="*80)
    print("💡 KHUYẾN NGHỊ")
    print("="*80)
    
    if php_count > 80 or java_count > 80:
        print("\n⚠️  Vấn đề phát hiện: PHP/Java xuất hiện quá nhiều!")
        print("\nNguyên nhân có thể:")
        print("  1. DATA có boilerplate text (ví dụ: 'Java/PHP/Python...' trong mọi job)")
        print("  2. PATTERNS quá rộng")
        print("  3. Đang search trong column không đúng")
        
        print("\nGiải pháp:")
        print("  Option 1: FILTER by job_group")
        print("    - Chỉ count PHP trong backend/fullstack jobs")
        print("    - Chỉ count Java trong backend/mobile jobs")
        
        print("\n  Option 2: STRICT patterns")
        print("    - Yêu cầu context rõ ràng (VD: 'PHP developer', 'Java programming')")
        print("    - Không match từ generic list")
        
        print("\n  Option 3: CLEAN data")
        print("    - Remove boilerplate text từ mo_ta_clean")
        print("    - Chỉ search trong job requirements, không search trong company intro")
    else:
        print("\n✅ Patterns trông OK!")
        print(f"   PHP: {php_count}%, Java: {java_count}%")
    
    # Export samples for manual review
    output_file = "data/processed/skill_diagnosis.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== PHP SAMPLES ===\n\n")
        for sample in php_samples:
            f.write(f"Job: {sample['title']}\n")
            f.write(f"Group: {sample['group']}\n")
            f.write(f"Context: {sample['context']}\n\n")
        
        f.write("\n=== JAVA SAMPLES ===\n\n")
        for sample in java_samples:
            f.write(f"Job: {sample['title']}\n")
            f.write(f"Group: {sample['group']}\n")
            f.write(f"Context: {sample['context']}\n\n")
    
    print(f"\n💾 Chi tiết đã xuất ra: {output_file}")
    print("\nReview file này để xem context cụ thể!")
    
    print("\n" + "="*80)
    print("✅ HOÀN TẤT CHẨN ĐOÁN")
    print("="*80)

if __name__ == "__main__":
    # Try different paths
    paths_to_try = [
        "data/processed/topcv_it_features.csv",
        "../data/processed/topcv_it_features.csv",
    ]
    
    for path in paths_to_try:
        try:
            diagnose_skill_patterns(path)
            break
        except FileNotFoundError:
            continue
    else:
        print("❌ Cannot find features CSV file!")
        print("   Make sure you're in the project root directory")