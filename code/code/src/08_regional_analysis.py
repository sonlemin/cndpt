#!/usr/bin/env python3
# src/08_regional_analysis.py
"""
Regional Analysis - Phân tích theo vùng miền

Analyses:
1. Job distribution by region/city
2. Salary comparison across regions
3. Skill demand by region
4. Company characteristics by region
5. Remote work prevalence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_PATH = "data/processed/topcv_it_features.csv"
FIG_DIR = Path("fig")
REPORTS_DIR = Path("reports")

# Create directories
FIG_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")


def load_data():
    """Load features data"""
    print("📂 Loading data...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"✅ Loaded {len(df)} jobs")
    
    # Check for regional features
    if 'region' not in df.columns and 'primary_location' not in df.columns:
        print("⚠️  No regional features found.")
        print("   Run feature extraction with regional features first.")
    
    return df


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
        print((region_dist / len(df) * 100).round(1))
        results['by_region'] = region_dist
    
    # 2. By city
    if 'primary_location' in df.columns:
        print("\n2. Top 10 Cities:")
        city_dist = df['primary_location'].value_counts().head(10)
        print(city_dist)
        results['by_city'] = city_dist
    
    # 3. Remote vs office
    if 'is_remote' in df.columns:
        print("\n3. Remote Work Distribution:")
        remote_dist = df['is_remote'].value_counts()
        print(remote_dist)
        
        remote_pct = df['is_remote'].mean() * 100
        print(f"\n📊 Remote work: {remote_pct:.1f}% of all jobs")
        results['remote'] = remote_dist
    
    # 4. Hybrid work
    if 'is_hybrid' in df.columns:
        hybrid_pct = df['is_hybrid'].mean() * 100
        print(f"📊 Hybrid work: {hybrid_pct:.1f}% of all jobs")
    
    return results


def analyze_regional_salary(df):
    """Salary comparison across regions"""
    print("\n" + "="*60)
    print("💰 REGIONAL SALARY ANALYSIS")
    print("="*60)
    
    if 'salary_avg' not in df.columns:
        print("⚠️  No salary data available")
        return None
    
    results = {}
    
    # 1. Salary by region
    if 'region' in df.columns:
        print("\n1. Average Salary by Region:")
        regional_salary = df.groupby('region')['salary_avg'].agg(['mean', 'median', 'std', 'count'])
        print(regional_salary.round(2))
        results['regional'] = regional_salary
        
        # Find highest/lowest
        max_region = regional_salary['mean'].idxmax()
        min_region = regional_salary['mean'].idxmin()
        diff_pct = (regional_salary.loc[max_region, 'mean'] - regional_salary.loc[min_region, 'mean']) / regional_salary.loc[min_region, 'mean'] * 100
        print(f"\n📊 {max_region} pays {diff_pct:.1f}% more than {min_region}")
    
    # 2. Salary by city (top cities)
    if 'primary_location' in df.columns:
        print("\n2. Average Salary by City (Top 10):")
        city_salary = df.groupby('primary_location')['salary_avg'].agg(['mean', 'median', 'count'])
        city_salary = city_salary[city_salary['count'] >= 5]  # Filter for cities with 5+ jobs
        city_salary_sorted = city_salary.sort_values('mean', ascending=False).head(10)
        print(city_salary_sorted.round(2))
        results['city'] = city_salary_sorted
    
    # 3. Remote vs office salary
    if 'is_remote' in df.columns:
        print("\n3. Remote vs Office Salary:")
        remote_salary = df.groupby('is_remote')['salary_avg'].agg(['mean', 'median'])
        remote_salary.index = ['Office', 'Remote']
        print(remote_salary.round(2))
        
        if len(remote_salary) == 2:
            diff = remote_salary.loc['Remote', 'mean'] - remote_salary.loc['Office', 'mean']
            print(f"\n📊 Remote jobs pay {'more' if diff > 0 else 'less'} (Δ{abs(diff):.1f}M)")
    
    return results


def analyze_regional_skills(df):
    """Skill demand by region"""
    print("\n" + "="*60)
    print("🔧 REGIONAL SKILLS ANALYSIS")
    print("="*60)
    
    if 'skills_detected' not in df.columns or 'region' not in df.columns:
        print("⚠️  Missing required columns")
        return None
    
    # Get top 15 skills overall
    all_skills = []
    for skills in df['skills_detected'].dropna():
        all_skills.extend([s.strip() for s in str(skills).split(',')])
    
    top_skills = pd.Series(all_skills).value_counts().head(15).index.tolist()
    print(f"\n📊 Top 15 skills being tracked: {', '.join(top_skills)}")
    
    # Skill demand by region
    skills_by_region = {}
    
    for region in df['region'].dropna().unique():
        region_df = df[df['region'] == region]
        
        skill_counts = {}
        for skill in top_skills:
            count = sum(
                1 for skills in region_df['skills_detected'].dropna()
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
    print("🏢 COMPANY ANALYSIS BY REGION")
    print("="*60)
    
    if 'company_size' not in df.columns or 'region' not in df.columns:
        print("⚠️  Missing company or region data")
        return None
    
    # Company size distribution by region
    print("\n1. Company Size Distribution by Region:")
    company_by_region = pd.crosstab(df['region'], df['company_size'], normalize='index') * 100
    print(company_by_region.round(1))
    
    # Foreign companies by region
    if 'is_foreign' in df.columns:
        print("\n2. Foreign Companies by Region (%):")
        foreign_by_region = df.groupby('region')['is_foreign'].mean() * 100
        print(foreign_by_region.round(1))
    
    # Product vs Outsourcing
    if 'is_product' in df.columns and 'is_outsourcing' in df.columns:
        print("\n3. Product vs Outsourcing by Region (%):")
        product_pct = df.groupby('region')['is_product'].mean() * 100
        outsourcing_pct = df.groupby('region')['is_outsourcing'].mean() * 100
        
        company_types = pd.DataFrame({
            'Product': product_pct,
            'Outsourcing': outsourcing_pct
        })
        print(company_types.round(1))
    
    return company_by_region


def visualize_regional_analysis(df):
    """Create regional visualizations"""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 1. Job distribution by city (top 10)
    if 'primary_location' in df.columns:
        city_counts = df['primary_location'].value_counts().head(10)
        colors = sns.color_palette("viridis", len(city_counts))
        axes[0].barh(range(len(city_counts)), city_counts.values, color=colors)
        axes[0].set_yticks(range(len(city_counts)))
        axes[0].set_yticklabels(city_counts.index)
        axes[0].set_title('Job Distribution by City (Top 10)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Jobs')
        axes[0].invert_yaxis()
        
        # Add values
        for i, v in enumerate(city_counts.values):
            axes[0].text(v, i, f' {v}', va='center')
    
    # 2. Average salary by region
    if 'salary_avg' in df.columns and 'region' in df.columns:
        regional_salary = df.groupby('region')['salary_avg'].mean().sort_values()
        colors = ['#ff6b6b' if val < regional_salary.median() else '#4ecdc4' for val in regional_salary.values]
        axes[1].bar(range(len(regional_salary)), regional_salary.values, color=colors)
        axes[1].set_xticks(range(len(regional_salary)))
        axes[1].set_xticklabels(regional_salary.index, rotation=45)
        axes[1].set_title('Average Salary by Region', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Average Salary (Million VND)')
        axes[1].axhline(regional_salary.median(), color='red', linestyle='--', alpha=0.5, label='Median')
        axes[1].legend()
        
        # Add values
        for i, v in enumerate(regional_salary.values):
            axes[1].text(i, v, f'{v:.1f}M', ha='center', va='bottom')
    
    # 3. Remote work percentage by region
    if 'is_remote' in df.columns and 'region' in df.columns:
        remote_by_region = df.groupby('region')['is_remote'].mean() * 100
        colors = sns.color_palette("coolwarm", len(remote_by_region))
        axes[2].bar(range(len(remote_by_region)), remote_by_region.values, color=colors)
        axes[2].set_xticks(range(len(remote_by_region)))
        axes[2].set_xticklabels(remote_by_region.index, rotation=45)
        axes[2].set_title('Remote Work Percentage by Region', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('% of Jobs')
        axes[2].set_ylim(0, 100)
        
        # Add values
        for i, v in enumerate(remote_by_region.values):
            axes[2].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    # 4. Skills heatmap by region
    if 'skills_detected' in df.columns and 'region' in df.columns:
        # Get top 10 skills
        all_skills = []
        for skills in df['skills_detected'].dropna():
            all_skills.extend([s.strip() for s in str(skills).split(',')])
        top_skills = pd.Series(all_skills).value_counts().head(10).index.tolist()
        
        # Count by region
        regions = df['region'].dropna().unique()
        skill_matrix = pd.DataFrame(index=regions, columns=top_skills, dtype=float)
        
        for region in regions:
            region_df = df[df['region'] == region]
            for skill in top_skills:
                count = sum(
                    1 for skills in region_df['skills_detected'].dropna()
                    if skill in str(skills)
                )
                skill_matrix.loc[region, skill] = count
        
        # Normalize by region (percentage)
        skill_matrix_pct = skill_matrix.div(skill_matrix.sum(axis=1), axis=0) * 100
        
        sns.heatmap(skill_matrix_pct.astype(float), annot=True, fmt='.0f', cmap='YlGnBu', 
                    ax=axes[3], cbar_kws={'label': '% within region'})
        axes[3].set_title('Skill Distribution by Region (%)', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('')
        axes[3].set_ylabel('')
    
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
    
    # Regional distribution
    if distribution and 'by_region' in distribution:
        distribution['by_region'].to_csv(REPORTS_DIR / 'regional_distribution.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'regional_distribution.csv'}")
    
    # City distribution
    if distribution and 'by_city' in distribution:
        distribution['by_city'].to_csv(REPORTS_DIR / 'city_distribution.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'city_distribution.csv'}")
    
    # Regional salary
    if salary_analysis and 'regional' in salary_analysis:
        salary_analysis['regional'].to_csv(REPORTS_DIR / 'regional_salary.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'regional_salary.csv'}")
    
    # City salary
    if salary_analysis and 'city' in salary_analysis:
        salary_analysis['city'].to_csv(REPORTS_DIR / 'city_salary.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'city_salary.csv'}")


def main():
    """Main analysis function"""
    print("\n" + "="*60)
    print("🗺️  REGIONAL ANALYSIS - IT JOB MARKET")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Run analyses
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
    print(f"   Figures: {FIG_DIR}/")
    print(f"   Reports: {REPORTS_DIR}/")


if __name__ == "__main__":
    main()