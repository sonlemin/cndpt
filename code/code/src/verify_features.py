#!/usr/bin/env python3
"""
verify_features.py

Script kiểm tra chi tiết features CSV để tìm bug

USAGE:
    cd ~/workspace/github.com/sonlemin/cndpt/code/code
    python src/verify_features.py
"""

import pandas as pd

def verify_features(csv_path):
    """Verify features CSV for bugs"""
    print("="*80)
    print("🔍 KIỂM TRA FEATURES CSV")
    print("="*80)
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return
    
    print(f"\n📊 Basic info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Check skills_str column
    if "skills_str" not in df.columns:
        print("\n❌ ERROR: 'skills_str' column not found!")
        return
    
    print(f"\n" + "="*80)
    print("📌 KIỂM TRA SKILLS_STR COLUMN")
    print("="*80)
    
    # Sample 10 jobs
    print(f"\n📄 Sample 10 jobs:")
    print("-"*80)
    
    for idx in range(min(10, len(df))):
        row = df.iloc[idx]
        
        title = row.get('tieu_de', 'N/A')[:60]
        group = row.get('job_group', 'N/A')
        skills = row.get('skills_str', 'N/A')
        
        print(f"\n{idx+1}. {title}")
        print(f"   Group: {group}")
        print(f"   Skills: {skills}")
    
    # Check if all rows have same skills
    print(f"\n" + "="*80)
    print("🔍 CHECK: Tất cả jobs có cùng skills?")
    print("="*80)
    
    unique_skills = df['skills_str'].unique()
    print(f"\nNumber of unique skill combinations: {len(unique_skills)}")
    
    if len(unique_skills) == 1:
        print(f"\n❌ BUG DETECTED: All jobs have identical skills!")
        print(f"   Skills: {unique_skills[0][:200]}...")
        print(f"\n🔧 FIX NEEDED: Script đang assign tất cả skills vào tất cả jobs!")
    else:
        print(f"\n✅ Good: Jobs have different skill combinations")
    
    # Count skill frequency
    print(f"\n" + "="*80)
    print("📊 SKILL FREQUENCY")
    print("="*80)
    
    # Explode skills
    all_skills = []
    for skills_str in df['skills_str'].dropna():
        if skills_str and skills_str != '':
            skills = skills_str.split(',')
            all_skills.extend(skills)
    
    from collections import Counter
    skill_counts = Counter(all_skills)
    
    print(f"\nTop 10 skills:")
    for skill, count in skill_counts.most_common(10):
        pct = count / len(df) * 100
        print(f"  {skill:20s}: {count:4d}/{len(df)} ({pct:5.1f}%)")
        
        if pct > 95:
            print(f"    ⚠️  WARNING: >95% - Quá cao!")
    
    # Check n_skills distribution
    if 'n_skills' in df.columns:
        print(f"\n" + "="*80)
        print("📊 N_SKILLS DISTRIBUTION")
        print("="*80)
        
        print(f"\nn_skills statistics:")
        print(df['n_skills'].describe())
        
        # Check if all have same n_skills
        unique_n = df['n_skills'].unique()
        if len(unique_n) == 1:
            print(f"\n❌ BUG: All jobs have n_skills = {unique_n[0]}")
        else:
            print(f"\n✅ Good: n_skills varies from {df['n_skills'].min()} to {df['n_skills'].max()}")
    
    # Diagnosis
    print(f"\n" + "="*80)
    print("💡 CHẨN ĐOÁN")
    print("="*80)
    
    if len(unique_skills) == 1:
        print("\n❌ CRITICAL BUG FOUND!")
        print("\nVấn đề: Tất cả jobs có cùng skills")
        print("\nNguyên nhân có thể:")
        print("  1. Script đang dùng wrong config (skills_str được hardcode)")
        print("  2. Bug trong extract_skills() function")
        print("  3. Skills được assign từ list tổng thể thay vì extract từ mỗi job")
        
        print("\nGiải pháp:")
        print("  1. Check code của 04_extract_features_improved.py")
        print("  2. Đảm bảo skills được extract từng job một")
        print("  3. Re-run extraction")
    
    elif skill_counts.most_common(1)[0][1] / len(df) > 0.95:
        print("\n⚠️  POTENTIAL BUG!")
        print(f"\nTop skill xuất hiện >95% jobs: {skill_counts.most_common(1)[0]}")
        print("\nCó thể:")
        print("  1. Pattern quá rộng")
        print("  2. Boilerplate text trong data")
        print("  3. Bug trong extraction")
    
    else:
        print("\n✅ Skills extraction trông OK!")
        print(f"   Unique combinations: {len(unique_skills)}")
        print(f"   Top skill frequency: {skill_counts.most_common(1)[0][1] / len(df) * 100:.1f}%")
    
    print(f"\n" + "="*80)

if __name__ == "__main__":
    verify_features("data/processed/topcv_it_features.csv")