#!/usr/bin/env python3
# src/07_temporal_analysis.py
"""
Temporal Analysis - Phân tích theo thời gian

Analyses:
1. Job posting trends over time (monthly, quarterly, yearly)
2. Seasonal patterns and decomposition
3. Day-of-week patterns
4. Skill demand evolution over time
5. Salary trends over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import statsmodels for seasonal decomposition
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    print("⚠️  statsmodels not installed. Seasonal decomposition will be skipped.")
    print("   Install with: pip install statsmodels")
    HAS_STATSMODELS = False

# Configuration
FEATURES_PATH = "data/processed/topcv_it_features.csv"
FIG_DIR = Path("fig")
REPORTS_DIR = Path("reports")

# Create directories
FIG_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data():
    """Load features data"""
    print("📂 Loading data...")
    df = pd.read_csv(FEATURES_PATH)
    
    # Convert posted_date to datetime if exists
    if 'posted_date' in df.columns:
        df['posted_date'] = pd.to_datetime(df['posted_date'], errors='coerce')
    
    print(f"✅ Loaded {len(df)} jobs")
    print(f"📅 Date range: {df['posted_date'].min()} to {df['posted_date'].max()}")
    
    return df


def analyze_posting_trends(df):
    """Analyze job posting trends"""
    print("\n" + "="*60)
    print("📊 POSTING TRENDS ANALYSIS")
    print("="*60)
    
    results = {}
    
    # Check if temporal features exist
    if 'posted_month' not in df.columns:
        print("⚠️  No temporal features found. Run feature extraction with temporal features first.")
        return results
    
    # 1. Monthly trends
    print("\n1. Monthly Trends:")
    monthly = df.groupby('posted_month').size()
    print(monthly)
    results['monthly'] = monthly
    
    # 2. Quarterly trends
    print("\n2. Quarterly Trends:")
    if 'posted_quarter' in df.columns:
        quarterly = df.groupby('posted_quarter').size()
        print(quarterly)
        results['quarterly'] = quarterly
    
    # 3. Yearly trends (if multiple years)
    if 'posted_year' in df.columns and df['posted_year'].nunique() > 1:
        print("\n3. Year-over-Year:")
        yearly = df.groupby('posted_year').size()
        print(yearly)
        
        # YoY growth
        yoy_growth = yearly.pct_change() * 100
        print(f"\nYoY Growth: {yoy_growth.values[-1]:.1f}%")
        results['yearly'] = yearly
        results['yoy_growth'] = yoy_growth
    
    # 4. Day of week patterns
    if 'posted_day_of_week' in df.columns:
        print("\n4. Day of Week Patterns:")
        dow = df['posted_day_of_week'].value_counts()
        print(dow.sort_index())
        results['day_of_week'] = dow
    
    return results


def analyze_seasonal_patterns(df):
    """Seasonal decomposition"""
    print("\n" + "="*60)
    print("🔄 SEASONAL PATTERN ANALYSIS")
    print("="*60)
    
    if not HAS_STATSMODELS:
        print("⚠️  Skipping seasonal decomposition (statsmodels not available)")
        return None
    
    if 'posted_date' not in df.columns or df['posted_date'].isna().all():
        print("⚠️  No posted_date data available")
        return None
    
    # Prepare time series (weekly)
    df_clean = df.dropna(subset=['posted_date'])
    ts = df_clean.set_index('posted_date').resample('W').size()
    
    if len(ts) < 14:  # Need at least 2 periods
        print(f"⚠️  Not enough data points ({len(ts)}). Need at least 14 weeks.")
        return None
    
    try:
        # Seasonal decomposition
        period = min(52, len(ts) // 2)  # Annual cycle or half the data
        decomposition = seasonal_decompose(ts, model='additive', period=period)
        
        print(f"✅ Seasonal decomposition completed (period={period} weeks)")
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
        }
    except Exception as e:
        print(f"⚠️  Seasonal decomposition failed: {e}")
        return None


def analyze_skill_evolution(df):
    """Skill demand over time"""
    print("\n" + "="*60)
    print("🔧 SKILL EVOLUTION ANALYSIS")
    print("="*60)
    
    if 'skills_detected' not in df.columns or 'posted_quarter' not in df.columns:
        print("⚠️  Missing required columns")
        return None
    
    # Get top 10 skills overall
    all_skills = []
    for skills in df['skills_detected'].dropna():
        all_skills.extend([s.strip() for s in str(skills).split(',')])
    
    top_skills = pd.Series(all_skills).value_counts().head(10).index.tolist()
    print(f"\n📊 Tracking evolution of top 10 skills: {', '.join(top_skills)}")
    
    # Count skills by quarter
    skills_by_quarter = {}
    
    for quarter in sorted(df['posted_quarter'].dropna().unique()):
        quarter_df = df[df['posted_quarter'] == quarter]
        
        skill_counts = {}
        for skill in top_skills:
            count = sum(
                1 for skills in quarter_df['skills_detected'].dropna()
                if skill in str(skills)
            )
            skill_counts[skill] = count
        
        skills_by_quarter[quarter] = skill_counts
    
    # Display results
    print("\nSkill Demand by Quarter:")
    skills_df = pd.DataFrame(skills_by_quarter).T
    print(skills_df)
    
    return skills_df


def analyze_salary_trends(df):
    """Salary trends over time"""
    print("\n" + "="*60)
    print("💰 SALARY TRENDS ANALYSIS")
    print("="*60)
    
    if 'salary_avg' not in df.columns or 'posted_month' not in df.columns:
        print("⚠️  Missing required columns")
        return None
    
    # Monthly salary trends
    salary_by_month = df.groupby('posted_month')['salary_avg'].agg(['mean', 'median', 'count'])
    print("\nMonthly Salary Trends:")
    print(salary_by_month)
    
    # Calculate growth
    if len(salary_by_month) > 1:
        first_salary = salary_by_month['mean'].iloc[0]
        last_salary = salary_by_month['mean'].iloc[-1]
        growth = (last_salary - first_salary) / first_salary * 100
        print(f"\n📈 Salary growth: {growth:.1f}% (from {first_salary:.1f}M to {last_salary:.1f}M)")
    
    return salary_by_month


def visualize_temporal_patterns(df, seasonal_decomp=None):
    """Create temporal visualizations"""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    # Determine number of subplots
    n_plots = 4 if seasonal_decomp else 3
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 1. Monthly trend
    if 'posted_month' in df.columns:
        monthly = df.groupby('posted_month').size()
        axes[0].plot(monthly.index, monthly.values, marker='o', linewidth=2, markersize=8)
        axes[0].set_title('Job Postings by Month', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Number of Jobs')
        axes[0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(monthly.index, monthly.values, 1)
        p = np.poly1d(z)
        axes[0].plot(monthly.index, p(monthly.index), "--", alpha=0.5, color='red', 
                     label=f'Trend: {"↑" if z[0] > 0 else "↓"}')
        axes[0].legend()
    
    # 2. Quarterly comparison
    if 'posted_quarter' in df.columns:
        quarterly = df.groupby('posted_quarter').size()
        colors = sns.color_palette("husl", len(quarterly))
        axes[1].bar(range(len(quarterly)), quarterly.values, color=colors)
        axes[1].set_xticks(range(len(quarterly)))
        axes[1].set_xticklabels(quarterly.index, rotation=0)
        axes[1].set_title('Job Postings by Quarter', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Quarter')
        axes[1].set_ylabel('Number of Jobs')
        
        # Add values on bars
        for i, v in enumerate(quarterly.values):
            axes[1].text(i, v, str(v), ha='center', va='bottom')
    
    # 3. Day of week pattern
    if 'posted_day_of_week' in df.columns:
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow = df['posted_day_of_week'].value_counts().reindex(dow_order, fill_value=0)
        
        colors = ['#ff6b6b' if d in ['Saturday', 'Sunday'] else '#4ecdc4' for d in dow_order]
        axes[2].bar(range(7), dow.values, color=colors)
        axes[2].set_xticks(range(7))
        axes[2].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[2].set_title('Job Postings by Day of Week', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Number of Jobs')
        axes[2].grid(True, alpha=0.3, axis='y')
    
    # 4. Heatmap: Month vs Year or Skill Evolution
    if 'posted_year' in df.columns and df['posted_year'].nunique() > 1:
        heatmap_data = df.pivot_table(
            values='tieu_de', 
            index='posted_month', 
            columns='posted_year', 
            aggfunc='count',
            fill_value=0
        )
        sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='YlOrRd', ax=axes[3], cbar_kws={'label': 'Jobs'})
        axes[3].set_title('Job Postings Heatmap (Month vs Year)', fontsize=14, fontweight='bold')
    else:
        # Alternative: Salary trend
        if 'salary_avg' in df.columns and 'posted_month' in df.columns:
            salary_trend = df.groupby('posted_month')['salary_avg'].mean()
            axes[3].plot(salary_trend.index, salary_trend.values, marker='o', color='green', linewidth=2)
            axes[3].set_title('Average Salary Trend', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Month')
            axes[3].set_ylabel('Average Salary (Million VND)')
            axes[3].grid(True, alpha=0.3)
        else:
            axes[3].text(0.5, 0.5, 'No data for additional plot', 
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].axis('off')
    
    plt.tight_layout()
    output_file = FIG_DIR / 'temporal_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    # Seasonal decomposition plot (if available)
    if seasonal_decomp and HAS_STATSMODELS:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        # Original
        axes[0].plot(seasonal_decomp['trend'].index, seasonal_decomp['trend'].index.map(
            lambda x: df[df['posted_date'] <= x].resample('W', on='posted_date').size().get(x, 0)
        ))
        axes[0].set_ylabel('Observed')
        axes[0].set_title('Seasonal Decomposition of Job Postings', fontsize=14, fontweight='bold')
        
        # Trend
        seasonal_decomp['trend'].plot(ax=axes[1])
        axes[1].set_ylabel('Trend')
        
        # Seasonal
        seasonal_decomp['seasonal'].plot(ax=axes[2])
        axes[2].set_ylabel('Seasonal')
        
        # Residual
        seasonal_decomp['residual'].plot(ax=axes[3])
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        
        plt.tight_layout()
        output_file = FIG_DIR / 'seasonal_decomposition.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()


def save_reports(trends, skill_evolution, salary_trends):
    """Save analysis reports to CSV"""
    print("\n" + "="*60)
    print("💾 SAVING REPORTS")
    print("="*60)
    
    # Monthly trends
    if 'monthly' in trends:
        trends['monthly'].to_csv(REPORTS_DIR / 'monthly_trends.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'monthly_trends.csv'}")
    
    # Quarterly summary
    if 'quarterly' in trends:
        trends['quarterly'].to_csv(REPORTS_DIR / 'quarterly_summary.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'quarterly_summary.csv'}")
    
    # Skill evolution
    if skill_evolution is not None:
        skill_evolution.to_csv(REPORTS_DIR / 'skill_evolution.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'skill_evolution.csv'}")
    
    # Salary trends
    if salary_trends is not None:
        salary_trends.to_csv(REPORTS_DIR / 'salary_trends.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'salary_trends.csv'}")


def main():
    """Main analysis function"""
    print("\n" + "="*60)
    print("🕐 TEMPORAL ANALYSIS - IT JOB MARKET")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Run analyses
    trends = analyze_posting_trends(df)
    seasonal_decomp = analyze_seasonal_patterns(df)
    skill_evolution = analyze_skill_evolution(df)
    salary_trends = analyze_salary_trends(df)
    
    # Visualizations
    visualize_temporal_patterns(df, seasonal_decomp)
    
    # Save reports
    save_reports(trends, skill_evolution, salary_trends)
    
    print("\n" + "="*60)
    print("✅ TEMPORAL ANALYSIS COMPLETED")
    print("="*60)
    print(f"\n📁 Outputs:")
    print(f"   Figures: {FIG_DIR}/")
    print(f"   Reports: {REPORTS_DIR}/")


if __name__ == "__main__":
    main()