#!/usr/bin/env python3
"""
05_eda_visualize_v2.py

EDA Visualization - VERSION 2

IMPROVEMENTS from v1:
- ✅ Limit experience charts to 10 years (better visibility)
- ✅ Filter outliers in experience data
- ✅ Better chart formatting
- ✅ Add data validation

USAGE:
    cd ~/workspace/github.com/sonlemin/cndpt/code/code
    cp ~/Downloads/05_eda_visualize_v2.py src/
    python src/05_eda_visualize_v2.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Config
try:
    from config_v2 import FEATURES_PATH, FIG_DIR
except ImportError:
    try:
        from config_improved import FEATURES_PATH, FIG_DIR
    except ImportError:
        from config import FEATURES_PATH, FIG_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def create_job_group_chart(df, output_dir):
    """Chart 1: Job group distribution"""
    print("📊 Creating job group chart...")
    
    plt.figure(figsize=(14, 6))
    
    # Count and sort
    job_counts = df["job_group"].value_counts()
    
    # Create color palette (highlight "other" in red)
    colors = ['#d62728' if x == 'other' else '#1f77b4' for x in job_counts.index]
    
    # Plot
    ax = sns.barplot(x=job_counts.index, y=job_counts.values, palette=colors)
    
    # Add count labels on bars
    for i, (idx, val) in enumerate(job_counts.items()):
        ax.text(i, val + 2, str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title("Phân bố nhóm vị trí (job_group)", fontsize=16, fontweight='bold')
    plt.xlabel("Job Group", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    
    # Save
    plt.savefig(f"{output_dir}/01_job_group_count.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 01_job_group_count.png")
    
    # Print stats
    total = len(df)
    other_count = job_counts.get('other', 0)
    other_pct = other_count / total * 100
    
    print(f"  📊 Statistics:")
    print(f"     'other' group: {other_count}/{total} ({other_pct:.1f}%)")
    
    if other_pct < 20:
        print(f"     ✅ Target achieved! (<20%)")
    else:
        print(f"     ⚠️  Still above 20%. Need further improvement.")

def create_skills_chart(df, output_dir):
    """Chart 2: Top skills"""
    print("📊 Creating skills chart...")
    
    plt.figure(figsize=(12, 8))
    
    # Explode skills
    skill_series = df["skills_str"].dropna().str.split(",").explode()
    skill_series = skill_series[skill_series != ""]
    
    if len(skill_series) == 0:
        print("  ⚠️  No skills data found")
        return
    
    # Top 20
    top_skills = skill_series.value_counts().head(20)
    
    # Plot horizontal bar
    ax = top_skills.sort_values().plot(kind="barh", color='steelblue', figsize=(12, 8))
    
    # Add count labels
    for i, (skill, count) in enumerate(top_skills.sort_values().items()):
        ax.text(count + 5, i, str(count), va='center', fontsize=10, fontweight='bold')
    
    plt.title("Top 20 kỹ năng phổ biến", fontsize=16, fontweight='bold')
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Skill", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/02_top_skills.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 02_top_skills.png")
    
    # Check if top skills are 100%
    total_jobs = len(df)
    for skill, count in top_skills.head(3).items():
        pct = count / total_jobs * 100
        if pct > 95:
            print(f"  ⚠️  Warning: '{skill}' appears in {pct:.0f}% jobs (may indicate over-broad pattern)")

def create_salary_charts(df, output_dir):
    """Chart 3-5: Salary analysis"""
    print("📊 Creating salary charts...")
    
    # Chart 3: Pie chart with explanation
    salary_counts = df["has_salary"].value_counts().reindex([0, 1], fill_value=0)
    
    plt.figure(figsize=(8, 8))
    
    if salary_counts[1] == 0:
        # No salary data - show explanation
        plt.text(0.5, 0.5, 
                "⚠️  100% jobs không có thông tin lương\n\n"
                "Đây là vấn đề DATA SOURCE:\n"
                "TopCV hiếm khi công khai lương\n"
                "(thường ghi 'Thỏa thuận')",
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        plt.axis('off')
    else:
        plt.pie(
            salary_counts,
            labels=["Không có lương", "Có lương"],
            autopct="%1.1f%%",
            startangle=90,
            colors=['#ff9999', '#66b3ff'],
            textprops={'fontsize': 14}
        )
        plt.title("Tỷ lệ tin có công khai lương", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_salary_pie.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 03_salary_pie.png")
    
    # Skip charts 4 & 5 if no salary data
    df_salary = df[df["salary_avg"].notna()].copy()
    if len(df_salary) < 10:
        print(f"  ⚠️  Insufficient salary data. Skipping charts 4 & 5.")

def create_experience_charts(df, output_dir):
    """Chart 6-7: Experience analysis with 10-year limit"""
    print("📊 Creating experience charts (limited to 10 years)...")
    
    if "experience_years" not in df.columns:
        print("  ⚠️  No experience_years column. Skipping.")
        return
    
    df_exp = df[df["experience_years"].notna()].copy()
    
    if len(df_exp) < 10:
        print(f"  ⚠️  Insufficient experience data ({len(df_exp)} jobs). Skipping.")
        return
    
    # ===== FILTER: Limit to 10 years =====
    df_exp_filtered = df_exp[df_exp["experience_years"] <= 10].copy()
    n_filtered = len(df_exp) - len(df_exp_filtered)
    
    if n_filtered > 0:
        print(f"  ℹ️  Filtered {n_filtered} outliers (>10 years experience)")
    
    # Chart 6: Experience distribution (0-10 years)
    plt.figure(figsize=(12, 6))
    
    # Histogram with better bins
    bins = np.arange(0, 11, 0.5)  # 0, 0.5, 1, 1.5, ..., 10
    plt.hist(df_exp_filtered["experience_years"], bins=bins, color='mediumseagreen', edgecolor='black', alpha=0.7)
    
    # Add median and mean lines
    median_val = df_exp_filtered["experience_years"].median()
    mean_val = df_exp_filtered["experience_years"].mean()
    
    plt.axvline(median_val, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median_val:.1f} năm')
    plt.axvline(mean_val, color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.1f} năm')
    
    plt.title("Phân bố yêu cầu kinh nghiệm (0-10 năm)", fontsize=16, fontweight='bold')
    plt.xlabel("Số năm kinh nghiệm", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xlim(0, 10)  # FIXED: Limit x-axis to 10 years
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/06_experience_hist.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 06_experience_hist.png (0-10 years)")
    
    # Chart 7: Experience by job group (0-10 years)
    plt.figure(figsize=(14, 7))
    
    # Only groups with enough data
    group_counts = df_exp_filtered["job_group"].value_counts()
    groups_to_plot = group_counts[group_counts >= 5].index.tolist()
    
    if len(groups_to_plot) > 0:
        df_plot = df_exp_filtered[df_exp_filtered["job_group"].isin(groups_to_plot)]
        
        # Sort by median
        group_order = df_plot.groupby("job_group")["experience_years"].median().sort_values(ascending=False).index
        
        # Boxplot
        ax = sns.boxplot(data=df_plot, x="job_group", y="experience_years", order=group_order,
                        palette="Set2", showfliers=True)
        
        plt.title("Yêu cầu kinh nghiệm theo nhóm vị trí (0-10 năm)", fontsize=16, fontweight='bold')
        plt.xlabel("Job Group", fontsize=12)
        plt.ylabel("Kinh nghiệm (năm)", fontsize=12)
        plt.ylim(0, 10)  # FIXED: Limit y-axis to 10 years
        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/07_experience_by_group.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: 07_experience_by_group.png (0-10 years)")
    else:
        print(f"  ⚠️  Not enough groups with data. Skipping chart 7.")
    
    # Print statistics
    print(f"\n  📊 Experience Statistics:")
    print(f"     Total jobs with experience: {len(df_exp)}")
    print(f"     Jobs with 0-10 years: {len(df_exp_filtered)} ({len(df_exp_filtered)/len(df_exp)*100:.1f}%)")
    print(f"     Median: {median_val:.1f} years")
    print(f"     Mean: {mean_val:.1f} years")
    print(f"     Min: {df_exp_filtered['experience_years'].min():.1f} years")
    print(f"     Max: {df_exp_filtered['experience_years'].max():.1f} years")

def create_correlation_matrix(df, output_dir):
    """Chart 8: Correlation matrix"""
    print("📊 Creating correlation matrix...")
    
    # Select numeric columns
    numeric_cols = []
    if "salary_avg" in df.columns:
        numeric_cols.append("salary_avg")
    if "experience_years" in df.columns:
        numeric_cols.append("experience_years")
    if "n_skills" in df.columns:
        numeric_cols.append("n_skills")
    
    if len(numeric_cols) < 2:
        print("  ⚠️  Not enough numeric columns. Skipping.")
        return
    
    df_numeric = df[numeric_cols].dropna()
    
    if len(df_numeric) < 10:
        print(f"  ⚠️  Insufficient data ({len(df_numeric)} rows). Skipping.")
        return
    
    plt.figure(figsize=(8, 7))
    corr = df_numeric.corr()
    
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    plt.title("Correlation Matrix", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/08_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 08_correlation_matrix.png")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("📊 EDA VISUALIZATION VERSION 2 (Experience limited to 10 years)")
    print("="*80)
    
    # Create output directory
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"\n📁 Output directory: {FIG_DIR}")
    
    # Load data
    print(f"\n📂 Loading data: {FEATURES_PATH}")
    try:
        df = pd.read_csv(FEATURES_PATH)
        print(f"✅ Loaded: {df.shape}")
    except FileNotFoundError:
        print(f"❌ File not found: {FEATURES_PATH}")
        print("   Please run 04_extract_features_improved.py first")
        return
    
    # Create charts
    print("\n" + "-"*80)
    create_job_group_chart(df, FIG_DIR)
    
    print("-"*80)
    create_skills_chart(df, FIG_DIR)
    
    print("-"*80)
    create_salary_charts(df, FIG_DIR)
    
    print("-"*80)
    create_experience_charts(df, FIG_DIR)
    
    print("-"*80)
    create_correlation_matrix(df, FIG_DIR)
    
    # Summary
    print("\n" + "="*80)
    print("✅ ĐÃ XUẤT BIỂU ĐỒ VÀO:", FIG_DIR)
    print("="*80)
    
    print("\n📊 Charts created:")
    charts = [
        "01_job_group_count.png",
        "02_top_skills.png",
        "03_salary_pie.png",
        "06_experience_hist.png (NEW: 0-10 years)",
        "07_experience_by_group.png (NEW: 0-10 years)",
        "08_correlation_matrix.png",
    ]
    
    for chart in charts:
        filepath = f"{FIG_DIR}/{chart.split(' ')[0]}"
        if os.path.exists(filepath):
            print(f"  ✅ {chart}")
    
    print("\n💡 Key improvements:")
    print("  - Experience charts now limited to 0-10 years (better visibility)")
    print("  - Outliers filtered for cleaner visualization")
    print("  - Better chart formatting and labels")
    
    print("\n🎯 Next steps:")
    print("  - Check 'other' percentage in console output")
    print("  - Review experience charts (06, 07) - should look much better!")
    print("  - If 'other' still >20%, run analyze_other_jobs.py to investigate")

if __name__ == "__main__":
    main()