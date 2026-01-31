#!/usr/bin/env python3
# src/08_regional_analysis.py
"""
Regional Analysis - Phân tích theo vùng miền

Analyses:
1. Job distribution by region (North/Central/South)
2. Salary comparison across regions
3. Skill demand by region
4. Remote work by region
5. Company size distribution by region

Input: data/processed/topcv_it_features_with_regions.csv
       OR data/processed/topcv_it_features.csv (will use alternative analysis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration - Try regional file first, fallback to original
FEATURES_PATH_REGIONAL = "data/processed/topcv_it_features_with_regions.csv"
FEATURES_PATH_ORIGINAL = "data/processed/topcv_it_features.csv"
FIG_DIR = Path("fig")
REPORTS_DIR = Path("reports")

# Create directories
FIG_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")


def load_data():
    """Load features data - try regional version first"""
    print("📂 Loading data...")
    
    # Try regional file first
    if Path(FEATURES_PATH_REGIONAL).exists():
        df = pd.read_csv(FEATURES_PATH_REGIONAL)
        print(f"✅ Loaded {len(df)} jobs from: {FEATURES_PATH_REGIONAL}")
        return df, True
    
    # Fallback to original
    elif Path(FEATURES_PATH_ORIGINAL).exists():
        df = pd.read_csv(FEATURES_PATH_ORIGINAL)
        print(f"✅ Loaded {len(df)} jobs from: {FEATURES_PATH_ORIGINAL}")
        print(f"⚠️  Using original file (no regional features)")
        return df, False
    
    else:
        print(f"❌ File not found:")
        print(f"   - {FEATURES_PATH_REGIONAL}")
        print(f"   - {FEATURES_PATH_ORIGINAL}")
        print(f"\n   Please run:")
        print(f"   1. python3 src/04_extract_features.py --source topcv")
        print(f"   2. python3 src/04b_add_regional_features.py")
        exit(1)


def check_regional_features(df):
    """Check if regional features exist"""
    regional_cols = ['region', 'primary_location', 'is_remote', 'company_size']
    available = [col for col in regional_cols if col in df.columns]
    
    print("\n🗺️  Regional Features Check:")
    if not available:
        print("   ❌ No regional features found")
        return False
    else:
        print(f"   ✅ Found: {', '.join(available)}")
        return True


def analyze_regional_distribution(df):
    """Job distribution across regions"""
    print("\n" + "="*60)
    print("🗺️  REGIONAL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    results = {}
    
    # 1. By region
    if 'region' in df.columns:
        print("\n1. Jobs by Region:")
        region_dist = df['region'].value_counts()
        print(region_dist)
        
        # Percentage
        print("\nPercentage:")
        pct = (region_dist / len(df) * 100).round(1)
        for region, percentage in pct.items():
            print(f"   {region:10s}: {percentage:5.1f}%")
        
        results['by_region'] = region_dist
    
    # 2. By city
    if 'primary_location' in df.columns:
        print("\n2. Top 10 Cities:")
        city_dist = df['primary_location'].value_counts().head(10)
        for city, count in city_dist.items():
            print(f"   {city:15s}: {count:3d} jobs ({count/len(df)*100:5.1f}%)")
        results['by_city'] = city_dist
    
    # 3. Remote vs office
    if 'is_remote' in df.columns:
        print("\n3. Remote Work Distribution:")
        remote_count = df['is_remote'].sum()
        office_count = len(df) - remote_count
        
        print(f"   Remote: {remote_count:3d} jobs ({remote_count/len(df)*100:5.1f}%)")
        print(f"   Office: {office_count:3d} jobs ({office_count/len(df)*100:5.1f}%)")
        
        if 'is_hybrid' in df.columns:
            hybrid_count = df['is_hybrid'].sum()
            print(f"   Hybrid: {hybrid_count:3d} jobs ({hybrid_count/len(df)*100:5.1f}%)")
        
        results['remote'] = pd.Series({
            'Remote': remote_count,
            'Office': office_count
        })
    
    return results


def analyze_regional_salary(df):
    """Salary comparison across regions"""
    print("\n" + "="*60)
    print("💰 REGIONAL SALARY ANALYSIS")
    print("="*60)
    
    if 'salary_avg' not in df.columns:
        print("⚠️  No salary data available")
        return None
    
    if 'region' not in df.columns:
        print("⚠️  No region data available")
        return None
    
    results = {}
    
    # Salary by region
    print("\n1. Average Salary by Region:")
    regional_salary = df.groupby('region')['salary_avg'].agg(['mean', 'median', 'std', 'count'])
    
    print(f"\n{'Region':<10} {'Mean':>8} {'Median':>8} {'StdDev':>8} {'Count':>6}")
    print("-" * 50)
    for region, row in regional_salary.iterrows():
        print(f"{region:<10} {row['mean']:>8.1f} {row['median']:>8.1f} {row['std']:>8.1f} {int(row['count']):>6}")
    
    results['regional'] = regional_salary
    
    # Find highest/lowest
    if len(regional_salary) > 1:
        max_region = regional_salary['mean'].idxmax()
        min_region = regional_salary['mean'].idxmin()
        max_salary = regional_salary.loc[max_region, 'mean']
        min_salary = regional_salary.loc[min_region, 'mean']
        diff = max_salary - min_salary
        diff_pct = (diff / min_salary) * 100
        
        print(f"\n📊 Salary Gap:")
        print(f"   Highest: {max_region} - {max_salary:.1f}M VND")
        print(f"   Lowest:  {min_region} - {min_salary:.1f}M VND")
        print(f"   Gap:     {diff:.1f}M VND ({diff_pct:.1f}%)")
    
    # Salary by top cities
    if 'primary_location' in df.columns:
        print("\n2. Average Salary by City (Top 10):")
        city_salary = df.groupby('primary_location')['salary_avg'].agg(['mean', 'median', 'count'])
        city_salary = city_salary[city_salary['count'] >= 5]  # Filter for cities with 5+ jobs
        
        if len(city_salary) > 0:
            city_salary_sorted = city_salary.sort_values('mean', ascending=False).head(10)
            
            print(f"\n{'City':<15} {'Mean':>8} {'Median':>8} {'Count':>6}")
            print("-" * 45)
            for city, row in city_salary_sorted.iterrows():
                print(f"{city:<15} {row['mean']:>8.1f} {row['median']:>8.1f} {int(row['count']):>6}")
            
            results['city'] = city_salary_sorted
    
    # Remote vs office salary
    if 'is_remote' in df.columns:
        print("\n3. Remote vs Office Salary:")
        remote_salary = df.groupby('is_remote')['salary_avg'].agg(['mean', 'median', 'count'])
        
        print(f"\n{'Type':<10} {'Mean':>8} {'Median':>8} {'Count':>6}")
        print("-" * 40)
        for is_remote, row in remote_salary.iterrows():
            work_type = 'Remote' if is_remote else 'Office'
            print(f"{work_type:<10} {row['mean']:>8.1f} {row['median']:>8.1f} {int(row['count']):>6}")
        
        if len(remote_salary) == 2:
            remote_mean = remote_salary.loc[True, 'mean']
            office_mean = remote_salary.loc[False, 'mean']
            diff = remote_mean - office_mean
            print(f"\n   Remote jobs pay {abs(diff):.1f}M {'more' if diff > 0 else 'less'} on average")
    
    return results


def analyze_regional_skills(df):
    """Skill demand by region"""
    print("\n" + "="*60)
    print("🔧 REGIONAL SKILLS ANALYSIS")
    print("="*60)
    
    # Check for required columns
    skills_col = 'skills_str' if 'skills_str' in df.columns else ('skills' if 'skills' in df.columns else None)
    
    if not skills_col or 'region' not in df.columns:
        print("⚠️  Missing required columns")
        return None
    
    # Get top 10 skills overall
    all_skills = []
    for skills in df[skills_col].dropna():
        if skills_col == 'skills_str':
            all_skills.extend([s.strip() for s in str(skills).split(',') if s.strip()])
        else:
            try:
                import ast
                all_skills.extend(ast.literal_eval(str(skills)))
            except:
                all_skills.extend([s.strip() for s in str(skills).split(',') if s.strip()])
    
    top_skills = pd.Series(all_skills).value_counts().head(10).index.tolist()
    print(f"\nTracking top 10 skills: {', '.join(top_skills)}")
    
    # Skill demand by region
    skills_by_region = {}
    
    for region in sorted(df['region'].dropna().unique()):
        region_df = df[df['region'] == region]
        
        skill_counts = {}
        for skill in top_skills:
            count = sum(
                1 for skills in region_df[skills_col].dropna()
                if skill in str(skills)
            )
            skill_counts[skill] = count
        
        # Top 10 for this region
        top_regional = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        skills_by_region[region] = top_regional
        
        print(f"\n{region.upper()} - Top 10 Skills:")
        for i, (skill, count) in enumerate(top_regional, 1):
            pct = count / len(region_df) * 100
            print(f"  {i:2d}. {skill:15s} - {count:3d} jobs ({pct:5.1f}%)")
    
    return skills_by_region


def analyze_company_by_region(df):
    """Company characteristics by region"""
    print("\n" + "="*60)
    print("🏢 COMPANY SIZE BY REGION")
    print("="*60)
    
    if 'company_size' not in df.columns or 'region' not in df.columns:
        print("⚠️  Missing company_size or region data")
        return None
    
    # Company size distribution by region
    print("\n1. Company Size Distribution by Region (%):")
    company_by_region = pd.crosstab(df['region'], df['company_size'], normalize='index') * 100
    
    print("\n" + company_by_region.to_string())
    
    return company_by_region


def visualize_regional_analysis(df):
    """Create regional visualizations"""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # 1. Job distribution by region
    if 'region' in df.columns:
        region_counts = df['region'].value_counts()
        colors = sns.color_palette("Set2", len(region_counts))
        
        axes[plot_idx].bar(range(len(region_counts)), region_counts.values, color=colors)
        axes[plot_idx].set_xticks(range(len(region_counts)))
        axes[plot_idx].set_xticklabels(region_counts.index, rotation=0)
        axes[plot_idx].set_title('Job Distribution by Region', fontsize=14, fontweight='bold')
        axes[plot_idx].set_ylabel('Number of Jobs')
        
        # Add values on bars
        for i, v in enumerate(region_counts.values):
            axes[plot_idx].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Add percentages
        for i, v in enumerate(region_counts.values):
            pct = v / len(df) * 100
            axes[plot_idx].text(i, v/2, f'{pct:.1f}%', ha='center', va='center', 
                               fontsize=12, color='white', fontweight='bold')
        
        plot_idx += 1
    
    # 2. Average salary by region
    if 'salary_avg' in df.columns and 'region' in df.columns:
        regional_salary = df.groupby('region')['salary_avg'].mean().sort_values()
        colors = ['#ff6b6b' if val < regional_salary.median() else '#4ecdc4' 
                  for val in regional_salary.values]
        
        axes[plot_idx].bar(range(len(regional_salary)), regional_salary.values, color=colors)
        axes[plot_idx].set_xticks(range(len(regional_salary)))
        axes[plot_idx].set_xticklabels(regional_salary.index, rotation=0)
        axes[plot_idx].set_title('Average Salary by Region', fontsize=14, fontweight='bold')
        axes[plot_idx].set_ylabel('Average Salary (Million VND)')
        axes[plot_idx].axhline(regional_salary.median(), color='red', linestyle='--', 
                              alpha=0.5, linewidth=2, label='Median')
        axes[plot_idx].legend()
        
        # Add values on bars
        for i, v in enumerate(regional_salary.values):
            axes[plot_idx].text(i, v, f'{v:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        plot_idx += 1
    
    # 3. Top cities
    if 'primary_location' in df.columns:
        city_counts = df['primary_location'].value_counts().head(8)
        colors = sns.color_palette("viridis", len(city_counts))
        
        axes[plot_idx].barh(range(len(city_counts)), city_counts.values, color=colors)
        axes[plot_idx].set_yticks(range(len(city_counts)))
        axes[plot_idx].set_yticklabels(city_counts.index)
        axes[plot_idx].set_title('Top 8 Cities by Job Count', fontsize=14, fontweight='bold')
        axes[plot_idx].set_xlabel('Number of Jobs')
        axes[plot_idx].invert_yaxis()
        
        # Add values
        for i, v in enumerate(city_counts.values):
            axes[plot_idx].text(v, i, f' {v}', va='center', fontweight='bold')
        
        plot_idx += 1
    
    # 4. Remote work by region OR company size
    if 'is_remote' in df.columns and 'region' in df.columns:
        remote_by_region = df.groupby('region')['is_remote'].mean() * 100
        colors = sns.color_palette("coolwarm", len(remote_by_region))
        
        axes[plot_idx].bar(range(len(remote_by_region)), remote_by_region.values, color=colors)
        axes[plot_idx].set_xticks(range(len(remote_by_region)))
        axes[plot_idx].set_xticklabels(remote_by_region.index, rotation=0)
        axes[plot_idx].set_title('Remote Work Percentage by Region', fontsize=14, fontweight='bold')
        axes[plot_idx].set_ylabel('% of Jobs')
        axes[plot_idx].set_ylim(0, max(remote_by_region.values) * 1.2)
        
        # Add values
        for i, v in enumerate(remote_by_region.values):
            axes[plot_idx].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    output_file = FIG_DIR / 'regional_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def save_reports(distribution, salary_analysis, skills_by_region):
    """Save analysis reports"""
    print("\n" + "="*60)
    print("💾 SAVING REPORTS")
    print("="*60)
    
    saved_any = False
    
    # Regional distribution
    if distribution and 'by_region' in distribution:
        distribution['by_region'].to_csv(REPORTS_DIR / 'regional_distribution.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'regional_distribution.csv'}")
        saved_any = True
    
    # City distribution
    if distribution and 'by_city' in distribution:
        distribution['by_city'].to_csv(REPORTS_DIR / 'city_distribution.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'city_distribution.csv'}")
        saved_any = True
    
    # Regional salary
    if salary_analysis and 'regional' in salary_analysis:
        salary_analysis['regional'].to_csv(REPORTS_DIR / 'regional_salary.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'regional_salary.csv'}")
        saved_any = True
    
    # City salary
    if salary_analysis and 'city' in salary_analysis:
        salary_analysis['city'].to_csv(REPORTS_DIR / 'city_salary.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'city_salary.csv'}")
        saved_any = True
    
    if not saved_any:
        print("   ℹ️  No regional reports generated")


def main():
    """Main analysis function"""
    print("\n" + "="*60)
    print("🗺️  REGIONAL ANALYSIS - IT JOB MARKET")
    print("="*60)
    
    # Load data
    df, has_regional_file = load_data()
    
    # Check regional features
    has_regional = check_regional_features(df)
    
    if not has_regional:
        print("\n" + "="*60)
        print("⚠️  REGIONAL DATA NOT AVAILABLE")
        print("="*60)
        print("\nTo enable regional analysis, run:")
        print("  python3 src/04b_add_regional_features.py")
        print("\nThis will:")
        print("  - Detect regions from job data (North/Central/South)")
        print("  - Extract city/province information")
        print("  - Detect remote work flags")
        print("  - Estimate company sizes")
        exit(1)
    
    # Run regional analyses
    distribution = analyze_regional_distribution(df)
    salary_analysis = analyze_regional_salary(df)
    skills_by_region = analyze_regional_skills(df)
    company_analysis = analyze_company_by_region(df)
    
    # Visualizations
    visualize_regional_analysis(df)
    
    # Save reports
    save_reports(distribution, salary_analysis, skills_by_region)
    
    print("\n" + "="*60)
    print("✅ REGIONAL ANALYSIS COMPLETED")
    print("="*60)
    print(f"\n📁 Outputs:")
    print(f"   Figures: {FIG_DIR}/regional_analysis.png")
    print(f"   Reports: {REPORTS_DIR}/")


if __name__ == "__main__":
    main()