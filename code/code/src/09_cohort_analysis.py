#!/usr/bin/env python3
# src/09_cohort_analysis.py
"""
Cohort Analysis & Clustering - Phân tích nhóm

Analyses:
1. Company size cohorts
2. Experience level cohorts
3. Salary band cohorts
4. K-means clustering (5 clusters)

Note: Association rules moved to 06_association_rules.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import clustering libraries
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    print("⚠️  scikit-learn not installed. Clustering will be skipped.")
    print("   Install with: pip install scikit-learn")
    HAS_SKLEARN = False

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
    return df


def analyze_company_cohorts(df):
    """Analysis by company size"""
    print("\n" + "="*60)
    print("🏢 COMPANY SIZE COHORTS ANALYSIS")
    print("="*60)
    
    if 'company_size' not in df.columns:
        print("⚠️  No company_size data available")
        return None
    
    cohorts = {}
    
    for size in df['company_size'].dropna().unique():
        cohort_df = df[df['company_size'] == size]
        
        # Get top skills
        skills_list = []
        for skills in cohort_df['skills_detected'].dropna():
            skills_list.extend([s.strip() for s in str(skills).split(',')])
        top_skills = pd.Series(skills_list).value_counts().head(10).index.tolist()
        
        cohorts[size] = {
            'count': len(cohort_df),
            'avg_salary': cohort_df['salary_avg'].mean() if 'salary_avg' in df.columns else None,
            'top_skills': top_skills,
            'avg_benefits': cohort_df['benefits_score'].mean() if 'benefits_score' in df.columns else None,
        }
        
        print(f"\n{size.upper()} ({cohorts[size]['count']} jobs):")
        print(f"  Avg Salary: {cohorts[size]['avg_salary']:.1f}M" if cohorts[size]['avg_salary'] else "  Avg Salary: N/A")
        print(f"  Avg Benefits Score: {cohorts[size]['avg_benefits']:.1f}" if cohorts[size]['avg_benefits'] else "  Avg Benefits: N/A")
        print(f"  Top Skills: {', '.join(top_skills[:5])}")
    
    return cohorts


def analyze_experience_cohorts(df):
    """Analysis by experience level"""
    print("\n" + "="*60)
    print("👔 EXPERIENCE LEVEL COHORTS ANALYSIS")
    print("="*60)
    
    if 'seniority' not in df.columns:
        print("⚠️  No seniority data available")
        return None
    
    cohorts = {}
    
    seniority_order = ['intern', 'junior', 'mid', 'senior', 'lead', 'manager']
    
    for level in seniority_order:
        cohort_df = df[df['seniority'] == level]
        
        if len(cohort_df) == 0:
            continue
        
        cohorts[level] = {
            'count': len(cohort_df),
            'avg_salary': cohort_df['salary_avg'].mean() if 'salary_avg' in df.columns else None,
            'salary_range': {
                'min': cohort_df['salary_min'].mean() if 'salary_min' in df.columns else None,
                'max': cohort_df['salary_max'].mean() if 'salary_max' in df.columns else None,
            },
        }
        
        print(f"\n{level.upper()} ({cohorts[level]['count']} jobs):")
        if cohorts[level]['avg_salary']:
            print(f"  Avg Salary: {cohorts[level]['avg_salary']:.1f}M")
            if cohorts[level]['salary_range']['min']:
                print(f"  Salary Range: {cohorts[level]['salary_range']['min']:.1f}M - {cohorts[level]['salary_range']['max']:.1f}M")
    
    return cohorts


def analyze_salary_cohorts(df):
    """Analysis by salary bands"""
    print("\n" + "="*60)
    print("💰 SALARY BAND COHORTS ANALYSIS")
    print("="*60)
    
    if 'salary_avg' not in df.columns:
        print("⚠️  No salary data available")
        return None
    
    # Create salary bands
    df['salary_band'] = pd.cut(
        df['salary_avg'],
        bins=[0, 15, 25, 40, 100],
        labels=['Entry (< 15M)', 'Mid (15-25M)', 'Senior (25-40M)', 'Lead (40M+)']
    )
    
    cohorts = {}
    
    for band in df['salary_band'].dropna().unique():
        cohort_df = df[df['salary_band'] == band]
        
        cohorts[str(band)] = {
            'count': len(cohort_df),
            'avg_experience': cohort_df['experience_years_avg'].mean() if 'experience_years_avg' in df.columns else None,
            'avg_skills_count': cohort_df['skills_count'].mean() if 'skills_count' in df.columns else None,
        }
        
        print(f"\n{band} ({cohorts[str(band)]['count']} jobs):")
        if cohorts[str(band)]['avg_experience']:
            print(f"  Avg Experience: {cohorts[str(band)]['avg_experience']:.1f} years")
        if cohorts[str(band)]['avg_skills_count']:
            print(f"  Avg Skills: {cohorts[str(band)]['avg_skills_count']:.1f}")
    
    return cohorts


def perform_clustering(df, n_clusters=5):
    """K-means clustering analysis"""
    print("\n" + "="*60)
    print("🔍 K-MEANS CLUSTERING ANALYSIS")
    print("="*60)
    
    if not HAS_SKLEARN:
        print("⚠️  Skipping clustering (scikit-learn not available)")
        return None
    
    # Select features for clustering
    feature_cols = []
    if 'salary_avg' in df.columns:
        feature_cols.append('salary_avg')
    if 'experience_years_avg' in df.columns:
        feature_cols.append('experience_years_avg')
    if 'skills_count' in df.columns:
        feature_cols.append('skills_count')
    if 'benefits_score' in df.columns:
        feature_cols.append('benefits_score')
    
    if len(feature_cols) < 2:
        print("⚠️  Not enough numeric features for clustering")
        return None
    
    print(f"\n📊 Clustering on features: {', '.join(feature_cols)}")
    
    # Prepare data
    features = df[feature_cols].fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters
    print(f"\n✅ Created {n_clusters} clusters")
    
    for i in range(n_clusters):
        cluster_df = df[df['cluster'] == i]
        print(f"\nCluster {i} ({len(cluster_df)} jobs, {len(cluster_df)/len(df)*100:.1f}%):")
        
        for col in feature_cols:
            print(f"  Avg {col}: {cluster_df[col].mean():.2f}")
        
        # Top skills in cluster
        skills_list = []
        for skills in cluster_df['skills_detected'].dropna():
            skills_list.extend([s.strip() for s in str(skills).split(',')])
        if skills_list:
            top_skills = pd.Series(skills_list).value_counts().head(5).index.tolist()
            print(f"  Top Skills: {', '.join(top_skills)}")
    
    return df['cluster']


def visualize_cohort_analysis(df, clusters=None):
    """Create cohort visualizations"""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    n_plots = 3 if clusters is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    else:
        axes = list(axes)
    
    # 1. Salary by company size
    if 'company_size' in df.columns and 'salary_avg' in df.columns:
        company_salary = df.groupby('company_size')['salary_avg'].mean().sort_values()
        colors = sns.color_palette("coolwarm", len(company_salary))
        axes[0].barh(range(len(company_salary)), company_salary.values, color=colors)
        axes[0].set_yticks(range(len(company_salary)))
        axes[0].set_yticklabels(company_salary.index)
        axes[0].set_title('Average Salary by Company Size', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Average Salary (Million VND)')
        
        for i, v in enumerate(company_salary.values):
            axes[0].text(v, i, f' {v:.1f}M', va='center')
    
    # 2. Salary by seniority
    if 'seniority' in df.columns and 'salary_avg' in df.columns:
        seniority_order = ['intern', 'junior', 'mid', 'senior', 'lead', 'manager']
        seniority_salary = df.groupby('seniority')['salary_avg'].mean()
        seniority_salary = seniority_salary.reindex([s for s in seniority_order if s in seniority_salary.index])
        
        colors = sns.color_palette("viridis", len(seniority_salary))
        axes[1].bar(range(len(seniority_salary)), seniority_salary.values, color=colors)
        axes[1].set_xticks(range(len(seniority_salary)))
        axes[1].set_xticklabels([s.title() for s in seniority_salary.index], rotation=45)
        axes[1].set_title('Average Salary by Seniority', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Average Salary (Million VND)')
        
        for i, v in enumerate(seniority_salary.values):
            axes[1].text(i, v, f'{v:.1f}M', ha='center', va='bottom', fontsize=9)
    
    # 3. Cluster visualization (if available)
    if clusters is not None and len(axes) > 2:
        if 'salary_avg' in df.columns and 'experience_years_avg' in df.columns:
            for cluster in sorted(df['cluster'].unique()):
                cluster_df = df[df['cluster'] == cluster]
                axes[2].scatter(
                    cluster_df['experience_years_avg'],
                    cluster_df['salary_avg'],
                    label=f'Cluster {cluster}',
                    alpha=0.6,
                    s=50
                )
            
            axes[2].set_xlabel('Experience (years)')
            axes[2].set_ylabel('Salary (Million VND)')
            axes[2].set_title('Job Clusters (Experience vs Salary)', fontsize=12, fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = FIG_DIR / 'cohort_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    # Clustering visualization (separate plot if clusters available)
    if clusters is not None and 'salary_avg' in df.columns and 'skills_count' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for cluster in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster]
            ax.scatter(
                cluster_df['skills_count'],
                cluster_df['salary_avg'],
                label=f'Cluster {cluster} ({len(cluster_df)} jobs)',
                alpha=0.6,
                s=100
            )
        
        ax.set_xlabel('Number of Skills Required', fontsize=12)
        ax.set_ylabel('Salary (Million VND)', fontsize=12)
        ax.set_title('Job Clusters: Skills vs Salary', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = FIG_DIR / 'clustering_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()


def save_reports(company_cohorts, experience_cohorts, _unused=None):
    """Save analysis reports"""
    print("\n" + "="*60)
    print("💾 SAVING REPORTS")
    print("="*60)
    
    # Company cohorts
    if company_cohorts:
        company_df = pd.DataFrame(company_cohorts).T
        company_df.to_csv(REPORTS_DIR / 'company_cohorts.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'company_cohorts.csv'}")
    
    # Experience cohorts
    if experience_cohorts:
        experience_df = pd.DataFrame(experience_cohorts).T
        experience_df.to_csv(REPORTS_DIR / 'experience_cohorts.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'experience_cohorts.csv'}")


def main():
    """Main analysis function"""
    print("\n" + "="*60)
    print("👥 COHORT ANALYSIS & CLUSTERING - IT JOB MARKET")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Run analyses
    company_cohorts = analyze_company_cohorts(df)
    experience_cohorts = analyze_experience_cohorts(df)
    salary_cohorts = analyze_salary_cohorts(df)
    
    # Clustering
    clusters = perform_clustering(df, n_clusters=5)
    
    # Visualizations
    visualize_cohort_analysis(df, clusters)
    
    # Save reports
    save_reports(company_cohorts, experience_cohorts, None)
    
    print("\n" + "="*60)
    print("✅ COHORT ANALYSIS & CLUSTERING COMPLETED")
    print("="*60)
    print(f"\n📁 Outputs:")
    print(f"   Figures: {FIG_DIR}/")
    print(f"   Reports: {REPORTS_DIR}/")
    print(f"\n💡 Note: For skill association rules, run: python3 src/06_association_rules.py")


if __name__ == "__main__":
    main()