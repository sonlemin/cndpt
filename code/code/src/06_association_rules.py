#!/usr/bin/env python3
# src/06_association_rules.py
"""
Association Rules Mining - Phân tích quy tắc kết hợp

Analyses:
1. Frequent itemsets (skill combinations)
2. Association rules (skill → skill)
3. Skill co-occurrence patterns
4. Network visualization
5. Top rules by support, confidence, lift

FIXED ISSUES:
- Changed 'skills_detected' to 'skills_str' (matching 04_extract_features.py output)
- Improved error handling
- Better data validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import mlxtend for association rules
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    HAS_MLXTEND = True
except ImportError:
    print("❌ Error: mlxtend not installed!")
    print("   Install with: pip install mlxtend --break-system-packages")
    print("   Then run this script again.")
    exit(1)

# Try to import networkx for network visualization
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    print("⚠️  networkx not installed. Network visualization will be skipped.")
    print("   Install with: pip install networkx --break-system-packages")
    HAS_NETWORKX = False

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
    
    if not Path(FEATURES_PATH).exists():
        print(f"❌ File not found: {FEATURES_PATH}")
        print(f"   Please run: python3 src/04_extract_features.py --source topcv")
        exit(1)
    
    df = pd.read_csv(FEATURES_PATH)
    print(f"✅ Loaded {len(df)} jobs")
    
    # Check for required columns
    if 'skills_str' not in df.columns and 'skills' not in df.columns:
        print("❌ No skills columns found!")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Please run feature extraction first: python3 src/04_extract_features.py")
        exit(1)
    
    return df


def prepare_transactions(df):
    """Prepare skill transactions for association rules"""
    print("\n" + "="*60)
    print("📦 PREPARING SKILL TRANSACTIONS")
    print("="*60)
    
    # Check which skills column is available
    skills_column = None
    if 'skills_str' in df.columns:
        skills_column = 'skills_str'
    elif 'skills' in df.columns:
        skills_column = 'skills'
    else:
        print("❌ No skills column found")
        print(f"   Available columns: {list(df.columns)}")
        return None
    
    print(f"✅ Using column: {skills_column}")
    
    # Extract transactions (list of skill sets)
    transactions = []
    
    for skills in df[skills_column].dropna():
        if skills_column == 'skills_str':
            # Format: "Python, Java, MySQL"
            skill_list = [s.strip() for s in str(skills).split(',') if s.strip()]
        else:
            # Format: "['Python', 'Java', 'MySQL']" (string representation of list)
            try:
                # Try to parse as list
                import ast
                skill_list = ast.literal_eval(str(skills))
                if not isinstance(skill_list, list):
                    skill_list = [s.strip() for s in str(skills).split(',') if s.strip()]
            except:
                skill_list = [s.strip() for s in str(skills).split(',') if s.strip()]
        
        if skill_list:
            transactions.append(skill_list)
    
    if not transactions:
        print("❌ No skill transactions found")
        return None
    
    print(f"\n✅ Prepared {len(transactions)} transactions")
    
    # Statistics
    skill_counts = [len(t) for t in transactions]
    print(f"\nTransaction Statistics:")
    print(f"  Average skills per job: {np.mean(skill_counts):.1f}")
    print(f"  Min skills: {np.min(skill_counts)}")
    print(f"  Max skills: {np.max(skill_counts)}")
    print(f"  Median skills: {np.median(skill_counts):.0f}")
    
    # Most common individual skills
    all_skills = []
    for t in transactions:
        all_skills.extend(t)
    
    top_skills = pd.Series(all_skills).value_counts().head(20)
    print(f"\nTop 20 Individual Skills:")
    for i, (skill, count) in enumerate(top_skills.items(), 1):
        pct = count / len(transactions) * 100
        print(f"  {i:2d}. {skill:20s} - {count:3d} jobs ({pct:5.1f}%)")
    
    return transactions


def mine_frequent_itemsets(transactions, min_support=0.05):
    """Mine frequent itemsets"""
    print("\n" + "="*60)
    print("🔍 MINING FREQUENT ITEMSETS")
    print("="*60)
    
    print(f"\nParameters:")
    print(f"  Min Support: {min_support} ({min_support*100:.1f}%)")
    
    # Convert to one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    print(f"\nEncoded matrix shape: {df_encoded.shape}")
    print(f"Total unique skills: {len(te.columns_)}")
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        print(f"\n❌ No frequent itemsets found with min_support={min_support}")
        print(f"   Try reducing min_support (e.g., 0.03 or 0.02)")
        return None
    
    # Add itemset size
    frequent_itemsets['size'] = frequent_itemsets['itemsets'].apply(len)
    
    print(f"\n✅ Found {len(frequent_itemsets)} frequent itemsets")
    
    # Statistics by size
    print(f"\nItemsets by Size:")
    for size in sorted(frequent_itemsets['size'].unique()):
        count = len(frequent_itemsets[frequent_itemsets['size'] == size])
        print(f"  Size {size}: {count} itemsets")
    
    # Top itemsets by support
    print(f"\nTop 15 Frequent Itemsets (by support):")
    top_itemsets = frequent_itemsets.nlargest(15, 'support')
    
    for idx, row in top_itemsets.iterrows():
        skills = ', '.join(sorted(list(row['itemsets'])))
        print(f"\n  {skills}")
        print(f"    Support: {row['support']:.3f} ({row['support']*len(transactions):.0f} jobs)")
    
    return frequent_itemsets


def mine_association_rules(frequent_itemsets, transactions, min_confidence=0.3, min_lift=1.2):
    """Mine association rules"""
    print("\n" + "="*60)
    print("🔗 MINING ASSOCIATION RULES")
    print("="*60)
    
    print(f"\nParameters:")
    print(f"  Min Confidence: {min_confidence} ({min_confidence*100:.0f}%)")
    print(f"  Min Lift: {min_lift}")
    
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    if len(rules) == 0:
        print(f"\n❌ No rules found with min_confidence={min_confidence}")
        print(f"   Try reducing min_confidence (e.g., 0.2 or 0.1)")
        return None
    
    # Filter by lift
    rules = rules[rules['lift'] >= min_lift]
    
    if len(rules) == 0:
        print(f"\n❌ No rules found with min_lift={min_lift}")
        return None
    
    # Add rule size
    rules['antecedent_size'] = rules['antecedents'].apply(len)
    rules['consequent_size'] = rules['consequents'].apply(len)
    
    print(f"\n✅ Found {len(rules)} association rules")
    
    # Statistics
    print(f"\nRules Statistics:")
    print(f"  Average confidence: {rules['confidence'].mean():.3f}")
    print(f"  Average lift: {rules['lift'].mean():.2f}")
    print(f"  Max lift: {rules['lift'].max():.2f}")
    
    # Sort by different metrics
    print(f"\n" + "="*80)
    print(f"TOP 20 RULES BY LIFT (Strongest Associations)")
    print("="*80)
    
    top_rules = rules.nlargest(20, 'lift')
    
    for idx, row in top_rules.iterrows():
        antecedents = ', '.join(sorted(list(row['antecedents'])))
        consequents = ', '.join(sorted(list(row['consequents'])))
        
        print(f"\n{antecedents:40s} → {consequents}")
        print(f"  Support: {row['support']:.3f}  |  Confidence: {row['confidence']:.3f}  |  Lift: {row['lift']:.2f}")
        print(f"  Interpretation: Jobs with {antecedents} are {row['lift']:.1f}x more likely to require {consequents}")
    
    # Top rules by confidence
    print(f"\n" + "="*80)
    print(f"TOP 15 RULES BY CONFIDENCE (Most Certain)")
    print("="*80)
    
    top_conf = rules.nlargest(15, 'confidence')
    
    for idx, row in top_conf.iterrows():
        antecedents = ', '.join(sorted(list(row['antecedents'])))
        consequents = ', '.join(sorted(list(row['consequents'])))
        
        print(f"\n{antecedents:40s} → {consequents}")
        print(f"  Confidence: {row['confidence']:.1%}  |  Lift: {row['lift']:.2f}")
    
    return rules


def analyze_skill_pairs(rules):
    """Analyze most common skill pairs"""
    print("\n" + "="*60)
    print("👥 SKILL PAIR ANALYSIS")
    print("="*60)
    
    # Filter to single antecedent and consequent
    pairs = rules[(rules['antecedent_size'] == 1) & (rules['consequent_size'] == 1)].copy()
    
    if len(pairs) == 0:
        print("⚠️  No single skill → single skill rules found")
        return None
    
    print(f"\n✅ Found {len(pairs)} skill pairs")
    
    # Sort by lift
    top_pairs = pairs.nlargest(20, 'lift')
    
    print(f"\nTop 20 Skill Pairs (by lift):")
    print("="*80)
    
    for idx, row in top_pairs.iterrows():
        skill1 = list(row['antecedents'])[0]
        skill2 = list(row['consequents'])[0]
        
        print(f"\n{skill1:25s} ↔ {skill2}")
        print(f"  Lift: {row['lift']:.2f}  |  Confidence: {row['confidence']:.1%}  |  Support: {row['support']:.3f}")
    
    return pairs


def visualize_rules(rules, top_n=20):
    """Visualize association rules"""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 1. Scatter: Support vs Confidence (colored by Lift)
    top_rules = rules.nlargest(100, 'lift')  # Top 100 for better visualization
    
    scatter = axes[0].scatter(
        top_rules['support'],
        top_rules['confidence'],
        c=top_rules['lift'],
        s=100,
        alpha=0.6,
        cmap='viridis'
    )
    axes[0].set_xlabel('Support', fontsize=12)
    axes[0].set_ylabel('Confidence', fontsize=12)
    axes[0].set_title('Association Rules: Support vs Confidence\n(Color = Lift)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[0], label='Lift')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Top rules by lift
    top_lift = rules.nlargest(top_n, 'lift')
    
    # Create labels
    labels = []
    for _, row in top_lift.iterrows():
        ant = ', '.join(list(row['antecedents']))[:20]
        cons = ', '.join(list(row['consequents']))[:20]
        labels.append(f"{ant} → {cons}")
    
    y_pos = range(len(labels))
    axes[1].barh(y_pos, top_lift['lift'].values, color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel('Lift', fontsize=12)
    axes[1].set_title(f'Top {top_n} Rules by Lift', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # 3. Lift distribution
    axes[2].hist(rules['lift'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[2].axvline(rules['lift'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rules["lift"].mean():.2f}')
    axes[2].axvline(rules['lift'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {rules["lift"].median():.2f}')
    axes[2].set_xlabel('Lift', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Lift Distribution', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Confidence vs Lift
    axes[3].scatter(rules['confidence'], rules['lift'], alpha=0.5, s=50)
    axes[3].set_xlabel('Confidence', fontsize=12)
    axes[3].set_ylabel('Lift', fontsize=12)
    axes[3].set_title('Confidence vs Lift', fontsize=14, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(rules['confidence'], rules['lift'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(rules['confidence'].min(), rules['confidence'].max(), 100)
    axes[3].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
    axes[3].legend()
    
    plt.tight_layout()
    output_file = FIG_DIR / 'association_rules.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_skill_network(rules, top_n=30):
    """Create network graph of skill relationships"""
    print("\n" + "="*60)
    print("🕸️  CREATING SKILL NETWORK")
    print("="*60)
    
    if not HAS_NETWORKX:
        print("⚠️  Skipping network visualization (networkx not available)")
        print("   Install with: pip install networkx --break-system-packages")
        return None
    
    # Get top rules
    top_rules = rules.nlargest(top_n, 'lift')
    
    # Create graph
    G = nx.DiGraph()
    
    for _, row in top_rules.iterrows():
        for ant in row['antecedents']:
            for cons in row['consequents']:
                # Add edge with weight = lift
                if G.has_edge(ant, cons):
                    G[ant][cons]['weight'] += row['lift']
                else:
                    G.add_edge(ant, cons, weight=row['lift'])
    
    print(f"\n✅ Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes based on degree
    node_sizes = [G.degree(node) * 200 + 500 for node in G.nodes()]
    
    # Edge widths based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_weight * 5 for w in edge_weights]
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                          alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, 
                          arrows=True, arrowsize=20, edge_color='gray',
                          connectionstyle='arc3,rad=0.1', ax=ax)
    
    ax.set_title(f'Skill Relationship Network (Top {top_n} Rules)', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    output_file = FIG_DIR / 'skill_network.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    return G


def save_reports(frequent_itemsets, rules, pairs):
    """Save association rules reports"""
    print("\n" + "="*60)
    print("💾 SAVING REPORTS")
    print("="*60)
    
    # Frequent itemsets
    if frequent_itemsets is not None:
        # Convert frozensets to strings
        itemsets_export = frequent_itemsets.copy()
        itemsets_export['itemsets'] = itemsets_export['itemsets'].apply(lambda x: ', '.join(sorted(list(x))))
        itemsets_export.to_csv(REPORTS_DIR / 'frequent_itemsets.csv', index=False)
        print(f"✅ Saved: {REPORTS_DIR / 'frequent_itemsets.csv'}")
    
    # Association rules
    if rules is not None:
        # Convert frozensets to strings
        rules_export = rules.copy()
        rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
        
        # Select key columns
        columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 
                  'antecedent_size', 'consequent_size']
        rules_export = rules_export[columns]
        
        rules_export.to_csv(REPORTS_DIR / 'association_rules.csv', index=False)
        print(f"✅ Saved: {REPORTS_DIR / 'association_rules.csv'}")
    
    # Skill pairs
    if pairs is not None:
        pairs_export = pairs.copy()
        pairs_export['skill1'] = pairs_export['antecedents'].apply(lambda x: list(x)[0])
        pairs_export['skill2'] = pairs_export['consequents'].apply(lambda x: list(x)[0])
        
        pairs_export = pairs_export[['skill1', 'skill2', 'support', 'confidence', 'lift']]
        pairs_export.to_csv(REPORTS_DIR / 'skill_pairs.csv', index=False)
        print(f"✅ Saved: {REPORTS_DIR / 'skill_pairs.csv'}")


def main():
    """Main association rules mining function"""
    print("\n" + "="*60)
    print("🔗 ASSOCIATION RULES MINING - SKILL COMBINATIONS")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Prepare transactions
    transactions = prepare_transactions(df)
    
    if transactions is None or len(transactions) < 10:
        print("\n❌ Not enough transactions for analysis")
        return
    
    # Mine frequent itemsets
    frequent_itemsets = mine_frequent_itemsets(transactions, min_support=0.05)
    
    if frequent_itemsets is None:
        print("\n⚠️  Try adjusting min_support parameter")
        return
    
    # Mine association rules
    rules = mine_association_rules(frequent_itemsets, transactions, 
                                   min_confidence=0.3, min_lift=1.2)
    
    if rules is None:
        print("\n⚠️  Try adjusting min_confidence or min_lift parameters")
        return
    
    # Analyze skill pairs
    pairs = analyze_skill_pairs(rules)
    
    # Visualizations
    visualize_rules(rules, top_n=20)
    create_skill_network(rules, top_n=30)
    
    # Save reports
    save_reports(frequent_itemsets, rules, pairs)
    
    print("\n" + "="*60)
    print("✅ ASSOCIATION RULES MINING COMPLETED")
    print("="*60)
    print(f"\n📁 Outputs:")
    print(f"   Figures: {FIG_DIR}/")
    print(f"     - association_rules.png")
    print(f"     - skill_network.png")
    print(f"   Reports: {REPORTS_DIR}/")
    print(f"     - frequent_itemsets.csv")
    print(f"     - association_rules.csv")
    print(f"     - skill_pairs.csv")
    
    # Summary insights
    if rules is not None and len(rules) > 0:
        print(f"\n📊 Key Insights:")
        
        # Top rule
        top_rule = rules.nlargest(1, 'lift').iloc[0]
        ant = ', '.join(list(top_rule['antecedents']))
        cons = ', '.join(list(top_rule['consequents']))
        print(f"   Strongest association: {ant} → {cons} (lift: {top_rule['lift']:.2f})")
        
        # Most common itemset
        if frequent_itemsets is not None:
            top_itemset = frequent_itemsets.nlargest(1, 'support').iloc[0]
            skills = ', '.join(sorted(list(top_itemset['itemsets'])))
            print(f"   Most common combination: {skills} ({top_itemset['support']:.1%} of jobs)")
        
        # Average lift
        print(f"   Average lift: {rules['lift'].mean():.2f}")
        print(f"   Rules with lift > 2.0: {len(rules[rules['lift'] > 2.0])}")


if __name__ == "__main__":
    main()