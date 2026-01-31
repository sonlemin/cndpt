#!/usr/bin/env python3
# src/11_predictive_analysis.py
"""
Predictive Analytics - Dự đoán & Mô hình

Analyses:
1. Skill demand forecast (time series)
2. Salary prediction (machine learning)
3. Job posting volume forecast
4. Emerging skills detection
5. Anomaly detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    print("⚠️  scikit-learn not installed. ML models will be skipped.")
    print("   Install with: pip install scikit-learn")
    HAS_SKLEARN = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    print("⚠️  statsmodels not installed. Time series forecasting will be skipped.")
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
sns.set_palette("Set2")


def load_data():
    """Load features data"""
    print("📂 Loading data...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"✅ Loaded {len(df)} jobs")
    
    # Convert date if exists
    if 'posted_date' in df.columns:
        df['posted_date'] = pd.to_datetime(df['posted_date'], errors='coerce')
    
    return df


def forecast_posting_volume(df, periods=12):
    """Forecast job posting volume"""
    print("\n" + "="*60)
    print("📈 JOB POSTING VOLUME FORECAST")
    print("="*60)
    
    if not HAS_STATSMODELS:
        print("⚠️  Skipping forecast (statsmodels not available)")
        return None
    
    if 'posted_month' not in df.columns:
        print("⚠️  No posted_month data available")
        return None
    
    # Monthly job counts
    monthly_counts = df.groupby('posted_month').size()
    
    if len(monthly_counts) < 6:
        print(f"⚠️  Not enough data ({len(monthly_counts)} months). Need at least 6.")
        return None
    
    print(f"\n📊 Historical data: {len(monthly_counts)} months")
    print(f"📅 Forecasting next {periods} months")
    
    try:
        # ARIMA model
        model = ARIMA(monthly_counts, order=(1, 1, 1))
        fitted = model.fit()
        
        # Forecast
        forecast = fitted.forecast(steps=periods)
        
        # Confidence intervals
        forecast_obj = fitted.get_forecast(steps=periods)
        conf_int = forecast_obj.conf_int()
        
        print(f"\n✅ Forecast completed")
        print(f"\nNext {min(3, periods)} months forecast:")
        for i in range(min(3, periods)):
            month_num = monthly_counts.index[-1] + i + 1
            print(f"  Month {month_num}: {forecast.iloc[i]:.0f} jobs (range: {conf_int.iloc[i, 0]:.0f}-{conf_int.iloc[i, 1]:.0f})")
        
        return {
            'historical': monthly_counts,
            'forecast': forecast,
            'confidence_interval': conf_int,
        }
    
    except Exception as e:
        print(f"⚠️  Forecast failed: {e}")
        return None


def predict_salary(df):
    """Predict salary based on features"""
    print("\n" + "="*60)
    print("💰 SALARY PREDICTION MODEL")
    print("="*60)
    
    if not HAS_SKLEARN:
        print("⚠️  Skipping salary prediction (scikit-learn not available)")
        return None
    
    # Check required columns
    required_cols = ['salary_avg']
    if not all(col in df.columns for col in required_cols):
        print("⚠️  Missing required columns")
        return None
    
    # Prepare features
    feature_cols = []
    
    if 'experience_years_avg' in df.columns:
        feature_cols.append('experience_years_avg')
    if 'skills_count' in df.columns:
        feature_cols.append('skills_count')
    if 'benefits_score' in df.columns:
        feature_cols.append('benefits_score')
    
    # Encode categorical variables
    if 'is_remote' in df.columns:
        df['is_remote_encoded'] = df['is_remote'].astype(int)
        feature_cols.append('is_remote_encoded')
    
    if 'company_size' in df.columns:
        le = LabelEncoder()
        df['company_size_encoded'] = le.fit_transform(df['company_size'].fillna('unknown'))
        feature_cols.append('company_size_encoded')
    
    if 'region' in df.columns:
        le_region = LabelEncoder()
        df['region_encoded'] = le_region.fit_transform(df['region'].fillna('unknown'))
        feature_cols.append('region_encoded')
    
    if len(feature_cols) < 2:
        print("⚠️  Not enough features for prediction")
        return None
    
    print(f"\n📊 Features used: {', '.join(feature_cols)}")
    
    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df['salary_avg'].dropna()
    
    # Align X and y
    valid_idx = y.index
    X = X.loc[valid_idx]
    
    if len(X) < 50:
        print(f"⚠️  Not enough samples ({len(X)}). Need at least 50.")
        return None
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n✅ Model trained on {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    print(f"\nModel Performance:")
    print(f"  RMSE: {rmse:.2f} Million VND")
    print(f"  R² Score: {r2:.3f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in importance.iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.3f}")
    
    return {
        'model': model,
        'feature_importance': importance,
        'rmse': rmse,
        'r2': r2,
        'predictions': pd.DataFrame({'actual': y_test, 'predicted': y_pred}),
    }


def detect_emerging_skills(df):
    """Detect skills with increasing demand"""
    print("\n" + "="*60)
    print("🚀 EMERGING SKILLS DETECTION")
    print("="*60)
    
    if 'skills_detected' not in df.columns or 'posted_quarter' not in df.columns:
        print("⚠️  Missing required columns")
        return None
    
    # Get all unique skills
    all_skills_set = set()
    for skills in df['skills_detected'].dropna():
        all_skills_set.update([s.strip() for s in str(skills).split(',')])
    
    # Get quarters
    quarters = sorted(df['posted_quarter'].dropna().unique())
    
    if len(quarters) < 2:
        print(f"⚠️  Need at least 2 quarters. Found: {len(quarters)}")
        return None
    
    print(f"\n📊 Analyzing {len(all_skills_set)} skills across {len(quarters)} quarters")
    
    emerging = []
    
    for skill in all_skills_set:
        counts = []
        
        for quarter in quarters:
            quarter_df = df[df['posted_quarter'] == quarter]
            count = sum(
                1 for skills in quarter_df['skills_detected'].dropna()
                if skill in str(skills)
            )
            counts.append(count)
        
        # Calculate growth (only if skill appears in first period)
        if counts[0] > 0:
            growth_rate = (counts[-1] - counts[0]) / counts[0] * 100
            
            # Consider emerging if growth > 30% and appeared in at least 5 jobs
            if growth_rate > 30 and counts[-1] >= 5:
                emerging.append({
                    'skill': skill,
                    'initial_count': counts[0],
                    'final_count': counts[-1],
                    'growth_rate': growth_rate,
                    'trend': counts,
                })
    
    # Sort by growth rate
    emerging = sorted(emerging, key=lambda x: x['growth_rate'], reverse=True)
    
    print(f"\n✅ Found {len(emerging)} emerging skills (growth > 30%)")
    print(f"\nTop 15 Emerging Skills:")
    print("="*80)
    
    for i, skill_info in enumerate(emerging[:15], 1):
        print(f"\n{i:2d}. {skill_info['skill']}")
        print(f"    Growth: {skill_info['growth_rate']:+.1f}% ({skill_info['initial_count']} → {skill_info['final_count']} jobs)")
    
    return emerging


def detect_anomalies(df):
    """Detect unusual job postings"""
    print("\n" + "="*60)
    print("🔍 ANOMALY DETECTION")
    print("="*60)
    
    if not HAS_SKLEARN:
        print("⚠️  Skipping anomaly detection (scikit-learn not available)")
        return None
    
    # Prepare features
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
        print("⚠️  Not enough features for anomaly detection")
        return None
    
    print(f"\n📊 Analyzing anomalies using: {', '.join(feature_cols)}")
    
    # Prepare data
    features = df[feature_cols].fillna(0)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    predictions = iso_forest.fit_predict(features)
    
    # Mark anomalies
    df['is_anomaly'] = predictions == -1
    anomalies = df[df['is_anomaly']]
    
    print(f"\n✅ Detected {len(anomalies)} anomalies ({len(anomalies)/len(df)*100:.1f}%)")
    
    if len(anomalies) > 0:
        print(f"\nExample Anomalies:")
        print("="*80)
        
        display_cols = ['tieu_de'] + [col for col in feature_cols if col in df.columns]
        for i, row in anomalies.head(5).iterrows():
            print(f"\n{row['tieu_de'][:60]}")
            for col in feature_cols:
                if col in df.columns:
                    print(f"  {col}: {row[col]}")
    
    return anomalies


def visualize_predictions(forecast_result, salary_result, emerging_skills):
    """Create prediction visualizations"""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    n_plots = sum([forecast_result is not None, salary_result is not None, emerging_skills is not None])
    
    if n_plots == 0:
        print("⚠️  No data to visualize")
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = list(axes)
    
    plot_idx = 0
    
    # 1. Forecast plot
    if forecast_result:
        historical = forecast_result['historical']
        forecast = forecast_result['forecast']
        conf_int = forecast_result['confidence_interval']
        
        # Plot historical
        axes[plot_idx].plot(historical.index, historical.values, 'o-', label='Historical', linewidth=2)
        
        # Plot forecast
        forecast_months = range(historical.index[-1] + 1, historical.index[-1] + 1 + len(forecast))
        axes[plot_idx].plot(forecast_months, forecast.values, 's--', 
                           label='Forecast', color='red', linewidth=2)
        
        # Confidence interval
        axes[plot_idx].fill_between(forecast_months, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                                    alpha=0.2, color='red', label='95% CI')
        
        axes[plot_idx].set_title('Job Posting Forecast', fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel('Month')
        axes[plot_idx].set_ylabel('Number of Jobs')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # 2. Salary prediction plot
    if salary_result:
        predictions = salary_result['predictions']
        
        axes[plot_idx].scatter(predictions['actual'], predictions['predicted'], alpha=0.5)
        
        # Perfect prediction line
        min_val = min(predictions['actual'].min(), predictions['predicted'].min())
        max_val = max(predictions['actual'].max(), predictions['predicted'].max())
        axes[plot_idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axes[plot_idx].set_xlabel('Actual Salary (Million VND)')
        axes[plot_idx].set_ylabel('Predicted Salary (Million VND)')
        axes[plot_idx].set_title(f'Salary Prediction (R²={salary_result["r2"]:.3f})', fontsize=12, fontweight='bold')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # 3. Emerging skills
    if emerging_skills and len(emerging_skills) > 0:
        top_emerging = emerging_skills[:10]
        skills = [s['skill'] for s in top_emerging]
        growth = [s['growth_rate'] for s in top_emerging]
        
        colors = ['#2ecc71' if g > 50 else '#3498db' for g in growth]
        axes[plot_idx].barh(range(len(skills)), growth, color=colors)
        axes[plot_idx].set_yticks(range(len(skills)))
        axes[plot_idx].set_yticklabels(skills)
        axes[plot_idx].set_xlabel('Growth Rate (%)')
        axes[plot_idx].set_title('Top 10 Emerging Skills', fontsize=12, fontweight='bold')
        axes[plot_idx].invert_yaxis()
        
        # Add values
        for i, v in enumerate(growth):
            axes[plot_idx].text(v, i, f' {v:.0f}%', va='center')
        
        plot_idx += 1
    
    plt.tight_layout()
    output_file = FIG_DIR / 'predictive_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def save_reports(forecast_result, salary_result, emerging_skills, anomalies):
    """Save prediction reports"""
    print("\n" + "="*60)
    print("💾 SAVING REPORTS")
    print("="*60)
    
    # Forecast
    if forecast_result:
        forecast_df = pd.DataFrame({
            'month': range(forecast_result['historical'].index[-1] + 1, 
                          forecast_result['historical'].index[-1] + 1 + len(forecast_result['forecast'])),
            'forecast': forecast_result['forecast'].values,
            'lower_bound': forecast_result['confidence_interval'].iloc[:, 0].values,
            'upper_bound': forecast_result['confidence_interval'].iloc[:, 1].values,
        })
        forecast_df.to_csv(REPORTS_DIR / 'posting_forecast.csv', index=False)
        print(f"✅ Saved: {REPORTS_DIR / 'posting_forecast.csv'}")
    
    # Salary model
    if salary_result:
        salary_result['feature_importance'].to_csv(REPORTS_DIR / 'salary_model_features.csv', index=False)
        print(f"✅ Saved: {REPORTS_DIR / 'salary_model_features.csv'}")
    
    # Emerging skills
    if emerging_skills:
        emerging_df = pd.DataFrame(emerging_skills)
        emerging_df = emerging_df.drop('trend', axis=1)  # Remove trend column (list)
        emerging_df.to_csv(REPORTS_DIR / 'emerging_skills.csv', index=False)
        print(f"✅ Saved: {REPORTS_DIR / 'emerging_skills.csv'}")
    
    # Anomalies
    if anomalies is not None and len(anomalies) > 0:
        anomaly_cols = ['tieu_de', 'salary_avg', 'experience_years_avg', 'skills_count', 'benefits_score']
        anomaly_cols = [col for col in anomaly_cols if col in anomalies.columns]
        anomalies[anomaly_cols].to_csv(REPORTS_DIR / 'anomalies.csv', index=False)
        print(f"✅ Saved: {REPORTS_DIR / 'anomalies.csv'}")


def main():
    """Main prediction function"""
    print("\n" + "="*60)
    print("🔮 PREDICTIVE ANALYSIS - IT JOB MARKET")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Run predictions
    forecast_result = forecast_posting_volume(df, periods=12)
    salary_result = predict_salary(df)
    emerging_skills = detect_emerging_skills(df)
    anomalies = detect_anomalies(df)
    
    # Visualizations
    visualize_predictions(forecast_result, salary_result, emerging_skills)
    
    # Save reports
    save_reports(forecast_result, salary_result, emerging_skills, anomalies)
    
    print("\n" + "="*60)
    print("✅ PREDICTIVE ANALYSIS COMPLETED")
    print("="*60)
    print(f"\n📁 Outputs:")
    print(f"   Figures: {FIG_DIR}/")
    print(f"   Reports: {REPORTS_DIR}/")


if __name__ == "__main__":
    main()