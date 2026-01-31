#!/usr/bin/env python3
# src/10_create_dashboard.py
"""
Interactive Dashboard Creation

Creates interactive HTML dashboards using Plotly:
1. Time series with range slider
2. Regional distribution (map + charts)
3. Skill evolution (animated timeline)
4. Salary explorer (interactive scatter)
5. Cluster visualization (3D if available)
6. Association rules network
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    print("❌ Error: plotly not installed!")
    print("   Install with: pip install plotly")
    print("   Then run this script again.")
    exit(1)

# Configuration
FEATURES_PATH = "data/processed/topcv_it_features.csv"
FIG_DIR = Path("fig")
REPORTS_DIR = Path("reports")

# Create directories
FIG_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Colors
COLORS = px.colors.qualitative.Set2


def load_data():
    """Load features data"""
    print("📂 Loading data...")
    df = pd.read_csv(FEATURES_PATH)
    
    # Convert dates
    if 'posted_date' in df.columns:
        df['posted_date'] = pd.to_datetime(df['posted_date'], errors='coerce')
    
    print(f"✅ Loaded {len(df)} jobs")
    return df


def create_main_dashboard(df):
    """Create main overview dashboard"""
    print("\n" + "="*60)
    print("📊 CREATING MAIN DASHBOARD")
    print("="*60)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Job Postings Over Time',
            'Top 10 Skills',
            'Salary Distribution',
            'Regional Distribution'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "pie"}]
        ]
    )
    
    # 1. Time series (if temporal data available)
    if 'posted_date' in df.columns:
        # Daily counts
        daily = df.groupby('posted_date').size().reset_index(name='count')
        
        # 7-day rolling average
        daily['rolling'] = daily['count'].rolling(window=7, center=True).mean()
        
        fig.add_trace(
            go.Scatter(
                x=daily['posted_date'],
                y=daily['count'],
                mode='markers',
                name='Daily',
                marker=dict(size=4, color='lightblue'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily['posted_date'],
                y=daily['rolling'],
                mode='lines',
                name='7-day avg',
                line=dict(color='blue', width=2),
                showlegend=False
            ),
            row=1, col=1
        )
    elif 'posted_month' in df.columns:
        # Monthly counts
        monthly = df.groupby('posted_month').size().reset_index(name='count')
        
        fig.add_trace(
            go.Scatter(
                x=monthly['posted_month'],
                y=monthly['count'],
                mode='lines+markers',
                name='Monthly',
                line=dict(color='blue', width=2),
                marker=dict(size=8),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 2. Top skills
    if 'skills_detected' in df.columns:
        skills_list = []
        for skills in df['skills_detected'].dropna():
            skills_list.extend([s.strip() for s in str(skills).split(',')])
        
        top_skills = pd.Series(skills_list).value_counts().head(10)
        
        fig.add_trace(
            go.Bar(
                x=top_skills.values,
                y=top_skills.index,
                orientation='h',
                marker=dict(color=COLORS[:len(top_skills)]),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Salary distribution
    if 'salary_avg' in df.columns:
        salaries = df['salary_avg'].dropna()
        
        fig.add_trace(
            go.Histogram(
                x=salaries,
                nbinsx=30,
                marker=dict(color='green', opacity=0.7),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. Regional distribution
    if 'region' in df.columns:
        regional = df['region'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=regional.index,
                values=regional.values,
                marker=dict(colors=COLORS[:len(regional)]),
                showlegend=False
            ),
            row=2, col=2
        )
    elif 'primary_location' in df.columns:
        # Use top cities if no region
        cities = df['primary_location'].value_counts().head(5)
        
        fig.add_trace(
            go.Pie(
                labels=cities.index,
                values=cities.values,
                marker=dict(colors=COLORS[:len(cities)]),
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Jobs", row=1, col=1)
    
    fig.update_xaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="", row=1, col=2)
    
    fig.update_xaxes(title_text="Salary (Million VND)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    fig.update_layout(
        title_text="IT Job Market Dashboard - Overview",
        height=800,
        showlegend=False,
        template='plotly_white'
    )
    
    output_file = FIG_DIR / 'dashboard_main.html'
    fig.write_html(output_file)
    print(f"✅ Saved: {output_file}")
    
    return fig


def create_skill_evolution_dashboard(df):
    """Create interactive skill evolution timeline"""
    print("\n" + "="*60)
    print("🔧 CREATING SKILL EVOLUTION DASHBOARD")
    print("="*60)
    
    if 'skills_detected' not in df.columns or 'posted_month' not in df.columns:
        print("⚠️  Missing required columns")
        return None
    
    # Get top 10 skills
    skills_list = []
    for skills in df['skills_detected'].dropna():
        skills_list.extend([s.strip() for s in str(skills).split(',')])
    
    top_skills = pd.Series(skills_list).value_counts().head(10).index.tolist()
    
    # Count skills by month
    months = sorted(df['posted_month'].dropna().unique())
    
    fig = go.Figure()
    
    for skill in top_skills:
        counts = []
        
        for month in months:
            month_df = df[df['posted_month'] == month]
            count = sum(
                1 for skills in month_df['skills_detected'].dropna()
                if skill in str(skills)
            )
            counts.append(count)
        
        fig.add_trace(go.Scatter(
            x=months,
            y=counts,
            mode='lines+markers',
            name=skill,
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    fig.update_layout(
        title='Skill Demand Evolution (Interactive)',
        xaxis_title='Month',
        yaxis_title='Number of Jobs',
        height=600,
        hovermode='x unified',
        template='plotly_white'
    )
    
    output_file = FIG_DIR / 'dashboard_skills.html'
    fig.write_html(output_file)
    print(f"✅ Saved: {output_file}")
    
    return fig


def create_salary_explorer(df):
    """Create interactive salary explorer"""
    print("\n" + "="*60)
    print("💰 CREATING SALARY EXPLORER")
    print("="*60)
    
    if 'salary_avg' not in df.columns:
        print("⚠️  No salary data available")
        return None
    
    # Prepare data
    plot_df = df.dropna(subset=['salary_avg']).copy()
    
    # Default color by region if available
    color_col = None
    if 'region' in plot_df.columns:
        color_col = 'region'
    elif 'seniority' in plot_df.columns:
        color_col = 'seniority'
    elif 'company_size' in plot_df.columns:
        color_col = 'company_size'
    
    # Size by skills_count if available
    size_col = None
    if 'skills_count' in plot_df.columns:
        size_col = 'skills_count'
    
    # X-axis: experience
    x_col = None
    if 'experience_years_avg' in plot_df.columns:
        x_col = 'experience_years_avg'
    elif 'experience_min' in plot_df.columns:
        x_col = 'experience_min'
    
    if x_col is None:
        print("⚠️  No experience data for X-axis")
        return None
    
    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x=x_col,
        y='salary_avg',
        color=color_col,
        size=size_col if size_col else None,
        hover_data=['tieu_de'] if 'tieu_de' in plot_df.columns else None,
        labels={
            'salary_avg': 'Salary (Million VND)',
            x_col: 'Experience (years)',
            'skills_count': 'Skills Required',
        },
        title='Salary Explorer (Interactive)',
        template='plotly_white'
    )
    
    # Update layout
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='white')))
    
    fig.update_layout(
        height=600,
        hovermode='closest',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    
    output_file = FIG_DIR / 'dashboard_salary.html'
    fig.write_html(output_file)
    print(f"✅ Saved: {output_file}")
    
    return fig


def create_regional_dashboard(df):
    """Create regional analysis dashboard"""
    print("\n" + "="*60)
    print("🗺️  CREATING REGIONAL DASHBOARD")
    print("="*60)
    
    if 'region' not in df.columns and 'primary_location' not in df.columns:
        print("⚠️  No regional data available")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Jobs by City',
            'Average Salary by Region',
            'Remote Work Distribution',
            'Skills by Region'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "pie"}, {"type": "bar"}]
        ]
    )
    
    # 1. Jobs by city
    if 'primary_location' in df.columns:
        cities = df['primary_location'].value_counts().head(10)
        
        fig.add_trace(
            go.Bar(
                x=cities.values,
                y=cities.index,
                orientation='h',
                marker=dict(color=px.colors.sequential.Viridis),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 2. Salary by region
    if 'salary_avg' in df.columns and 'region' in df.columns:
        regional_salary = df.groupby('region')['salary_avg'].mean().sort_values()
        
        fig.add_trace(
            go.Bar(
                x=regional_salary.index,
                y=regional_salary.values,
                marker=dict(color=COLORS[:len(regional_salary)]),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Remote work
    if 'is_remote' in df.columns:
        remote = df['is_remote'].value_counts()
        labels = ['Office', 'Remote'] if len(remote) == 2 else remote.index
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=remote.values,
                marker=dict(colors=['#3498db', '#2ecc71']),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. Top skill by region
    if 'skills_detected' in df.columns and 'region' in df.columns:
        # Get most common skill per region
        region_top_skill = {}
        
        for region in df['region'].dropna().unique():
            region_df = df[df['region'] == region]
            skills_list = []
            
            for skills in region_df['skills_detected'].dropna():
                skills_list.extend([s.strip() for s in str(skills).split(',')])
            
            if skills_list:
                top = pd.Series(skills_list).value_counts().iloc[0]
                region_top_skill[region] = top
        
        if region_top_skill:
            regions = list(region_top_skill.keys())
            counts = list(region_top_skill.values())
            
            fig.add_trace(
                go.Bar(
                    x=regions,
                    y=counts,
                    marker=dict(color=COLORS[:len(regions)]),
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # Update layout
    fig.update_xaxes(title_text="Jobs", row=1, col=1)
    fig.update_xaxes(title_text="Region", row=1, col=2)
    fig.update_yaxes(title_text="Salary (M VND)", row=1, col=2)
    fig.update_xaxes(title_text="Region", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    fig.update_layout(
        title_text="Regional Analysis Dashboard",
        height=800,
        showlegend=False,
        template='plotly_white'
    )
    
    output_file = FIG_DIR / 'dashboard_regional.html'
    fig.write_html(output_file)
    print(f"✅ Saved: {output_file}")
    
    return fig


def create_cluster_dashboard(df):
    """Create cluster visualization dashboard"""
    print("\n" + "="*60)
    print("🔍 CREATING CLUSTER DASHBOARD")
    print("="*60)
    
    # Check if we have clustering results or need to cluster
    if 'cluster' not in df.columns:
        print("⚠️  No cluster data. Running clustering first...")
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
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
                print("⚠️  Not enough features for clustering")
                return None
            
            # Cluster
            features = df[feature_cols].fillna(0)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(features_scaled)
            
            print(f"✅ Created {5} clusters")
            
        except ImportError:
            print("⚠️  scikit-learn not available")
            return None
    
    # Create 3D scatter if we have 3+ features
    if all(col in df.columns for col in ['salary_avg', 'experience_years_avg', 'skills_count']):
        
        plot_df = df.dropna(subset=['salary_avg', 'experience_years_avg', 'skills_count']).copy()
        plot_df['cluster'] = plot_df['cluster'].astype(str)
        
        fig = px.scatter_3d(
            plot_df,
            x='experience_years_avg',
            y='skills_count',
            z='salary_avg',
            color='cluster',
            hover_data=['tieu_de'] if 'tieu_de' in plot_df.columns else None,
            labels={
                'experience_years_avg': 'Experience (years)',
                'skills_count': 'Skills Required',
                'salary_avg': 'Salary (M VND)',
                'cluster': 'Cluster'
            },
            title='Job Clusters (3D Interactive)',
            template='plotly_white'
        )
        
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        
        fig.update_layout(
            height=700,
            scene=dict(
                xaxis_title='Experience (years)',
                yaxis_title='Skills Required',
                zaxis_title='Salary (M VND)'
            )
        )
        
        output_file = FIG_DIR / 'dashboard_clusters.html'
        fig.write_html(output_file)
        print(f"✅ Saved: {output_file}")
        
        return fig
    
    # Fallback: 2D scatter
    elif all(col in df.columns for col in ['salary_avg', 'experience_years_avg']):
        
        plot_df = df.dropna(subset=['salary_avg', 'experience_years_avg']).copy()
        plot_df['cluster'] = plot_df['cluster'].astype(str)
        
        fig = px.scatter(
            plot_df,
            x='experience_years_avg',
            y='salary_avg',
            color='cluster',
            size='skills_count' if 'skills_count' in plot_df.columns else None,
            hover_data=['tieu_de'] if 'tieu_de' in plot_df.columns else None,
            labels={
                'experience_years_avg': 'Experience (years)',
                'salary_avg': 'Salary (M VND)',
                'cluster': 'Cluster'
            },
            title='Job Clusters (Interactive)',
            template='plotly_white'
        )
        
        fig.update_layout(height=600)
        
        output_file = FIG_DIR / 'dashboard_clusters.html'
        fig.write_html(output_file)
        print(f"✅ Saved: {output_file}")
        
        return fig
    
    else:
        print("⚠️  Not enough numeric features for visualization")
        return None


def create_comprehensive_dashboard(df):
    """Create single comprehensive dashboard with tabs"""
    print("\n" + "="*60)
    print("🎨 CREATING COMPREHENSIVE DASHBOARD")
    print("="*60)
    
    # This would require Dash or similar framework
    # For now, we create individual dashboards
    
    print("✅ Individual dashboards created. See fig/ directory.")
    print("\n📁 Dashboard files:")
    print("   - dashboard_main.html         (Overview)")
    print("   - dashboard_skills.html       (Skill evolution)")
    print("   - dashboard_salary.html       (Salary explorer)")
    print("   - dashboard_regional.html     (Regional analysis)")
    print("   - dashboard_clusters.html     (Cluster visualization)")


def main():
    """Main dashboard creation function"""
    print("\n" + "="*60)
    print("🎨 INTERACTIVE DASHBOARD CREATION")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Create dashboards
    create_main_dashboard(df)
    create_skill_evolution_dashboard(df)
    create_salary_explorer(df)
    create_regional_dashboard(df)
    create_cluster_dashboard(df)
    
    # Summary
    create_comprehensive_dashboard(df)
    
    print("\n" + "="*60)
    print("✅ DASHBOARD CREATION COMPLETED")
    print("="*60)
    print(f"\n📁 Outputs: {FIG_DIR}/")
    print("\n🌐 Open in browser:")
    print(f"   {FIG_DIR / 'dashboard_main.html'}")
    print(f"   {FIG_DIR / 'dashboard_skills.html'}")
    print(f"   {FIG_DIR / 'dashboard_salary.html'}")
    print(f"   {FIG_DIR / 'dashboard_regional.html'}")
    print(f"   {FIG_DIR / 'dashboard_clusters.html'}")
    print("\n💡 Tip: Double-click HTML files to open in browser")


if __name__ == "__main__":
    main()