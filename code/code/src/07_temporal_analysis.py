#!/usr/bin/env python3
# src/07_temporal_analysis.py
"""
Temporal Analysis - Phân tích theo thời gian

FIXED: Gracefully handle missing temporal data

Analyses:
1. Job posting trends over time (monthly, quarterly, yearly)
2. Seasonal patterns and decomposition
3. Day-of-week patterns
4. Skill demand evolution over time
5. Salary trends over time

NOTE: Requires temporal features (posted_date, posted_month, etc.)
      If not available, will skip temporal analysis
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
    print("   Install with: pip install statsmodels --break-system-packages")
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


def check_temporal_features(df):
    """Check if temporal features exist"""
    temporal_cols = ['posted_date', 'posted_month', 'posted_quarter', 'posted_year', 'posted_day_of_week']
    available = [col for col in temporal_cols if col in df.columns]
    
    print("\n📅 Temporal Features Check:")
    if not available:
        print("   ❌ No temporal features found")
        print(f"\n   Available columns: {list(df.columns)[:10]}...")
        return False
    else:
        print(f"   ✅ Found: {', '.join(available)}")
        return True


def load_data():
    """Load features data"""
    print("📂 Loading data...")
    
    if not Path(FEATURES_PATH).exists():
        print(f"❌ File not found: {FEATURES_PATH}")
        print(f"   Please run: python3 src/04_extract_features.py --source topcv")
        exit(1)
    
    df = pd.read_csv(FEATURES_PATH)
    print(f"✅ Loaded {len(df)} jobs")
    
    # Convert posted_date to datetime if exists
    if 'posted_date' in df.columns:
        df['posted_date'] = pd.to_datetime(df['posted_date'], errors='coerce')
        date_count = df['posted_date'].notna().sum()
        if date_count > 0:
            print(f"📅 Date range: {df['posted_date'].min()} to {df['posted_date'].max()}")
        else:
            print(f"⚠️  posted_date column exists but has no valid dates")
    
    return df


def analyze_posting_trends(df):
    """Analyze job posting trends"""
    print("\n" + "="*60)
    print("📊 POSTING TRENDS ANALYSIS")
    print("="*60)
    
    results = {}
    
    # Check if temporal features exist
    if 'posted_month' not in df.columns:
        print("⚠️  No temporal features found. Skipping trend analysis.")
        print("   To enable temporal analysis:")
        print("   1. Modify scraping scripts (01, 02) to extract posted_date")
        print("   2. Re-run feature extraction with temporal features")
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
    
    if len(df_clean) == 0:
        print("⚠️  No valid dates in posted_date column")
        return None
    
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
    
    # Check for required columns
    skills_col = 'skills_str' if 'skills_str' in df.columns else 'skills'
    
    if skills_col not in df.columns or 'posted_quarter' not in df.columns:
        print(f"⚠️  Missing required columns")
        print(f"   Need: {skills_col} and posted_quarter")
        
        if 'posted_quarter' not in df.columns:
            print("\n   ℹ️  Temporal features not available.")
            print("   To enable skill evolution analysis:")
            print("   1. Extract posted_date during scraping")
            print("   2. Re-run feature extraction")
        
        return None
    
    # Get top 10 skills overall
    all_skills = []
    for skills in df[skills_col].dropna():
        if skills_col == 'skills_str':
            all_skills.extend([s.strip() for s in str(skills).split(',')])
        else:
            try:
                import ast
                skill_list = ast.literal_eval(str(skills))
                all_skills.extend(skill_list)
            except:
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
                1 for skills in quarter_df[skills_col].dropna()
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
    
    if 'salary_avg' not in df.columns:
        print("⚠️  No salary_avg column found")
        return None
    
    if 'posted_month' not in df.columns:
        print("⚠️  No posted_month column found")
        print("   Cannot analyze salary trends without temporal data")
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


def create_alternative_analysis(df):
    """Create analysis without temporal data"""
    print("\n" + "="*60)
    print("📊 ALTERNATIVE ANALYSIS (No Temporal Data)")
    print("="*60)
    
    print("\nSince temporal data is not available, here are alternative insights:")
    
    # 1. Overall statistics
    print("\n1. Dataset Overview:")
    print(f"   Total jobs: {len(df):,}")
    
    if 'job_group' in df.columns:
        print(f"\n2. Job Distribution:")
        job_dist = df['job_group'].value_counts().head(5)
        for group, count in job_dist.items():
            print(f"   {group:15s}: {count:4d} ({count/len(df)*100:5.1f}%)")
    
    if 'salary_avg' in df.columns:
        salary_data = df['salary_avg'].dropna()
        if len(salary_data) > 0:
            print(f"\n3. Salary Statistics:")
            print(f"   Mean:   {salary_data.mean():.1f}M VND")
            print(f"   Median: {salary_data.median():.1f}M VND")
            print(f"   Min:    {salary_data.min():.1f}M VND")
            print(f"   Max:    {salary_data.max():.1f}M VND")
    
    skills_col = 'skills_str' if 'skills_str' in df.columns else ('skills' if 'skills' in df.columns else None)
    if skills_col:
        all_skills = []
        for skills in df[skills_col].dropna():
            if skills_col == 'skills_str':
                all_skills.extend([s.strip() for s in str(skills).split(',')])
            else:
                try:
                    import ast
                    all_skills.extend(ast.literal_eval(str(skills)))
                except:
                    all_skills.extend([s.strip() for s in str(skills).split(',')])
        
        if all_skills:
            print(f"\n4. Top 10 Skills:")
            top_skills = pd.Series(all_skills).value_counts().head(10)
            for i, (skill, count) in enumerate(top_skills.items(), 1):
                print(f"   {i:2d}. {skill:20s} - {count:3d} jobs ({count/len(df)*100:5.1f}%)")


def visualize_temporal_patterns(df, seasonal_decomp=None):
    """Create temporal visualizations"""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    # Check if we have temporal data
    has_temporal = 'posted_month' in df.columns or 'posted_date' in df.columns
    
    if not has_temporal:
        print("⚠️  No temporal data available for visualization")
        print("   Creating alternative visualizations instead...")
        create_alternative_visualizations(df)
        return
    
    # Determine number of subplots
    n_plots = 4 if seasonal_decomp else 3
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # 1. Monthly trend
    if 'posted_month' in df.columns:
        monthly = df.groupby('posted_month').size()
        axes[plot_idx].plot(monthly.index, monthly.values, marker='o', linewidth=2, markersize=8)
        axes[plot_idx].set_title('Job Postings by Month', fontsize=14, fontweight='bold')
        axes[plot_idx].set_xlabel('Month')
        axes[plot_idx].set_ylabel('Number of Jobs')
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(monthly.index, monthly.values, 1)
        p = np.poly1d(z)
        axes[plot_idx].plot(monthly.index, p(monthly.index), "--", alpha=0.5, color='red', 
                     label=f'Trend: {"↑" if z[0] > 0 else "↓"}')
        axes[plot_idx].legend()
        plot_idx += 1
    
    # 2. Quarterly comparison
    if 'posted_quarter' in df.columns:
        quarterly = df.groupby('posted_quarter').size()
        colors = sns.color_palette("husl", len(quarterly))
        axes[plot_idx].bar(range(len(quarterly)), quarterly.values, color=colors)
        axes[plot_idx].set_xticks(range(len(quarterly)))
        axes[plot_idx].set_xticklabels(quarterly.index, rotation=0)
        axes[plot_idx].set_title('Job Postings by Quarter', fontsize=14, fontweight='bold')
        axes[plot_idx].set_xlabel('Quarter')
        axes[plot_idx].set_ylabel('Number of Jobs')
        
        # Add values on bars
        for i, v in enumerate(quarterly.values):
            axes[plot_idx].text(i, v, str(v), ha='center', va='bottom')
        plot_idx += 1
    
    # 3. Day of week pattern
    if 'posted_day_of_week' in df.columns:
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow = df['posted_day_of_week'].value_counts().reindex(dow_order, fill_value=0)
        
        colors = ['#ff6b6b' if d in ['Saturday', 'Sunday'] else '#4ecdc4' for d in dow_order]
        axes[plot_idx].bar(range(7), dow.values, color=colors)
        axes[plot_idx].set_xticks(range(7))
        axes[plot_idx].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[plot_idx].set_title('Job Postings by Day of Week', fontsize=14, fontweight='bold')
        axes[plot_idx].set_ylabel('Number of Jobs')
        axes[plot_idx].grid(True, alpha=0.3, axis='y')
        plot_idx += 1
    
    # 4. Heatmap or Salary trend
    if plot_idx < 4:
        if 'posted_year' in df.columns and df['posted_year'].nunique() > 1:
            heatmap_data = df.pivot_table(
                values='tieu_de', 
                index='posted_month', 
                columns='posted_year', 
                aggfunc='count',
                fill_value=0
            )
            sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='YlOrRd', ax=axes[plot_idx], cbar_kws={'label': 'Jobs'})
            axes[plot_idx].set_title('Job Postings Heatmap (Month vs Year)', fontsize=14, fontweight='bold')
        elif 'salary_avg' in df.columns and 'posted_month' in df.columns:
            salary_trend = df.groupby('posted_month')['salary_avg'].mean()
            axes[plot_idx].plot(salary_trend.index, salary_trend.values, marker='o', color='green', linewidth=2)
            axes[plot_idx].set_title('Average Salary Trend', fontsize=14, fontweight='bold')
            axes[plot_idx].set_xlabel('Month')
            axes[plot_idx].set_ylabel('Average Salary (Million VND)')
            axes[plot_idx].grid(True, alpha=0.3)
        else:
            axes[plot_idx].text(0.5, 0.5, 'No data for additional plot', 
                        ha='center', va='center', transform=axes[plot_idx].transAxes)
            axes[plot_idx].axis('off')
    
    plt.tight_layout()
    output_file = FIG_DIR / 'temporal_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_alternative_visualizations(df):
    """Create visualizations when no temporal data"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 1. Job group distribution
    if 'job_group' in df.columns:
        job_dist = df['job_group'].value_counts().head(10)
        axes[0].barh(range(len(job_dist)), job_dist.values)
        axes[0].set_yticks(range(len(job_dist)))
        axes[0].set_yticklabels(job_dist.index)
        axes[0].set_title('Top 10 Job Groups', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Count')
        axes[0].invert_yaxis()
    else:
        axes[0].text(0.5, 0.5, 'No job_group data', ha='center', va='center')
        axes[0].axis('off')
    
    # 2. Salary distribution
    if 'salary_avg' in df.columns:
        salary_data = df['salary_avg'].dropna()
        axes[1].hist(salary_data, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(salary_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {salary_data.mean():.1f}M')
        axes[1].axvline(salary_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {salary_data.median():.1f}M')
        axes[1].set_title('Salary Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Salary (Million VND)')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No salary data', ha='center', va='center')
        axes[1].axis('off')
    
    # 3. Top skills
    skills_col = 'skills_str' if 'skills_str' in df.columns else ('skills' if 'skills' in df.columns else None)
    if skills_col:
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
        
        if all_skills:
            top_skills = pd.Series(all_skills).value_counts().head(15)
            axes[2].barh(range(len(top_skills)), top_skills.values)
            axes[2].set_yticks(range(len(top_skills)))
            axes[2].set_yticklabels(top_skills.index)
            axes[2].set_title('Top 15 Skills', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Count')
            axes[2].invert_yaxis()
        else:
            axes[2].text(0.5, 0.5, 'No skills data', ha='center', va='center')
            axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'No skills data', ha='center', va='center')
        axes[2].axis('off')
    
    # 4. Experience distribution
    if 'experience_years' in df.columns:
        exp_data = df['experience_years'].dropna()
        if len(exp_data) > 0:
            axes[3].hist(exp_data, bins=20, edgecolor='black', alpha=0.7)
            axes[3].axvline(exp_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {exp_data.mean():.1f}y')
            axes[3].axvline(exp_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {exp_data.median():.1f}y')
            axes[3].set_title('Experience Requirements', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Years of Experience')
            axes[3].set_ylabel('Frequency')
            axes[3].legend()
        else:
            axes[3].text(0.5, 0.5, 'No experience data', ha='center', va='center')
            axes[3].axis('off')
    else:
        axes[3].text(0.5, 0.5, 'No experience data', ha='center', va='center')
        axes[3].axis('off')
    
    plt.tight_layout()
    output_file = FIG_DIR / 'alternative_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def save_reports(trends, skill_evolution, salary_trends):
    """Save analysis reports to CSV"""
    print("\n" + "="*60)
    print("💾 SAVING REPORTS")
    print("="*60)
    
    # Monthly trends
    if trends and 'monthly' in trends:
        trends['monthly'].to_csv(REPORTS_DIR / 'monthly_trends.csv')
        print(f"✅ Saved: {REPORTS_DIR / 'monthly_trends.csv'}")
    
    # Quarterly summary
    if trends and 'quarterly' in trends:
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
    
    if not (trends or skill_evolution or salary_trends):
        print("   ℹ️  No temporal reports generated (missing temporal data)")


def main():
    """Main analysis function"""
    print("\n" + "="*60)
    print("🕐 TEMPORAL ANALYSIS - IT JOB MARKET")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Check temporal features
    has_temporal = check_temporal_features(df)
    
    if not has_temporal:
        print("\n" + "="*60)
        print("⚠️  TEMPORAL DATA NOT AVAILABLE")
        print("="*60)
        print("\nThis script requires temporal features like:")
        print("  - posted_date")
        print("  - posted_month")
        print("  - posted_quarter")
        print("  - posted_year")
        print("\nTo enable temporal analysis:")
        print("  1. Modify scraping scripts (01, 02) to extract posting dates")
        print("  2. Add temporal feature extraction to script 04")
        print("  3. Re-run the data pipeline")
        print("\nFor now, running alternative analysis...")
        
        # Run alternative analysis
        create_alternative_analysis(df)
        create_alternative_visualizations(df)
        
        print("\n" + "="*60)
        print("✅ ALTERNATIVE ANALYSIS COMPLETED")
        print("="*60)
        print(f"\n📁 Output: {FIG_DIR}/alternative_analysis.png")
        return
    
    # Run temporal analyses
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