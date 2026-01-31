#!/usr/bin/env python3
# src/05_merge_sources.py
"""
Merge TopCV and VietnamWorks Data

Combines job listings from both sources into a single dataset.
Handles deduplication, standardization, and validation.

Usage:
    python3 src/05_merge_sources.py
    python3 src/05_merge_sources.py --dedup-strategy fuzzy
    python3 src/05_merge_sources.py --similarity-threshold 0.9
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from difflib import SequenceMatcher
import re
import warnings
warnings.filterwarnings('ignore')

# Import from unified config
try:
    from config import SOURCES, MERGED_CLEAN_PATH, MERGED_FEATURES_PATH
except ImportError:
    print("⚠️  Config not found, using defaults")
    
    DATA_DIR = Path("data")
    PROCESSED_DIR = DATA_DIR / "processed"
    
    SOURCES = {
        'topcv': {
            'clean_path': PROCESSED_DIR / 'topcv_it_clean.csv',
            'features_path': PROCESSED_DIR / 'topcv_it_features.csv',
        },
        'vietnamworks': {
            'clean_path': PROCESSED_DIR / 'vietnamworks_it_clean.csv',
            'features_path': PROCESSED_DIR / 'vietnamworks_it_features.csv',
        }
    }
    
    MERGED_CLEAN_PATH = PROCESSED_DIR / 'merged_it_clean.csv'
    MERGED_FEATURES_PATH = PROCESSED_DIR / 'merged_it_features.csv'


class DataMerger:
    """Merge data from multiple job sources"""
    
    def __init__(self, similarity_threshold=0.85, dedup_strategy='fuzzy'):
        """
        Initialize merger
        
        Args:
            similarity_threshold: Title similarity threshold for dedup (0.0-1.0)
            dedup_strategy: 'exact', 'fuzzy', or 'both'
        """
        self.similarity_threshold = similarity_threshold
        self.dedup_strategy = dedup_strategy
        
        # Statistics
        self.stats = {
            'topcv_jobs': 0,
            'vnw_jobs': 0,
            'exact_duplicates': 0,
            'fuzzy_duplicates': 0,
            'total_duplicates': 0,
            'final_jobs': 0,
        }
    
    def normalize_title(self, title):
        """
        Normalize job title for deduplication
        
        Args:
            title: Job title string
        
        Returns:
            Normalized title
        """
        if pd.isna(title):
            return ""
        
        t = str(title).lower()
        
        # Remove common recruitment words
        words_to_remove = [
            'tuyển', 'cần', 'tìm', 'hiring', 'urgent',
            'gấp', 'ngay', 'immediately', 'needed', 'cần gấp',
        ]
        
        for word in words_to_remove:
            t = re.sub(rf'\b{word}\b', '', t)
        
        # Remove special chars, keep alphanumeric and spaces
        t = re.sub(r'[^\w\s]', ' ', t)
        
        # Normalize spaces
        t = ' '.join(t.split())
        
        return t.strip()
    
    def detect_exact_duplicates(self, df1, df2):
        """
        Detect exact title matches
        
        Args:
            df1: First dataframe
            df2: Second dataframe
        
        Returns:
            Set of indices in df2 that are exact duplicates
        """
        duplicates = set()
        
        # Normalize titles
        titles1_norm = df1['tieu_de'].apply(self.normalize_title)
        titles2_norm = df2['tieu_de'].apply(self.normalize_title)
        
        # Find exact matches
        titles1_set = set(titles1_norm)
        
        for idx, title2 in enumerate(titles2_norm):
            if title2 and title2 in titles1_set:
                duplicates.add(idx)
        
        return duplicates
    
    def detect_fuzzy_duplicates(self, df1, df2, existing_dups=None):
        """
        Detect fuzzy title matches using sequence matching
        
        Args:
            df1: First dataframe
            df2: Second dataframe
            existing_dups: Set of already-found duplicate indices to skip
        
        Returns:
            Set of indices in df2 that are fuzzy duplicates
        """
        if existing_dups is None:
            existing_dups = set()
        
        duplicates = set()
        
        # Normalize titles
        titles1_norm = df1['tieu_de'].apply(self.normalize_title).tolist()
        titles2_norm = df2['tieu_de'].apply(self.normalize_title).tolist()
        
        # Only check non-exact duplicates
        indices_to_check = [i for i in range(len(df2)) if i not in existing_dups]
        
        # Limit fuzzy matching to avoid excessive computation
        max_fuzzy_checks = min(500, len(indices_to_check))
        
        print(f"   Checking {max_fuzzy_checks} titles for fuzzy matches...")
        
        for idx2 in indices_to_check[:max_fuzzy_checks]:
            title2 = titles2_norm[idx2]
            if not title2:
                continue
            
            len2 = len(title2)
            
            for title1 in titles1_norm:
                if not title1:
                    continue
                
                len1 = len(title1)
                
                # Skip if length difference is too large
                if abs(len1 - len2) > 30:
                    continue
                
                # Calculate similarity
                similarity = SequenceMatcher(None, title1, title2).ratio()
                
                if similarity >= self.similarity_threshold:
                    duplicates.add(idx2)
                    break
        
        return duplicates
    
    def detect_duplicates(self, df1, df2):
        """
        Detect duplicates using selected strategy
        
        Args:
            df1: First dataframe (TopCV)
            df2: Second dataframe (VietnamWorks)
        
        Returns:
            Set of duplicate indices in df2
        """
        print(f"\n🔍 Detecting duplicates (strategy: {self.dedup_strategy})...")
        
        all_duplicates = set()
        
        # Exact matching
        if self.dedup_strategy in ['exact', 'both']:
            print(f"   Exact matching...")
            exact_dups = self.detect_exact_duplicates(df1, df2)
            all_duplicates.update(exact_dups)
            self.stats['exact_duplicates'] = len(exact_dups)
            print(f"   Found {len(exact_dups)} exact matches")
        
        # Fuzzy matching
        if self.dedup_strategy in ['fuzzy', 'both']:
            print(f"   Fuzzy matching (threshold: {self.similarity_threshold})...")
            fuzzy_dups = self.detect_fuzzy_duplicates(df1, df2, existing_dups=all_duplicates)
            new_fuzzy = fuzzy_dups - all_duplicates
            all_duplicates.update(fuzzy_dups)
            self.stats['fuzzy_duplicates'] = len(new_fuzzy)
            print(f"   Found {len(new_fuzzy)} fuzzy matches")
        
        self.stats['total_duplicates'] = len(all_duplicates)
        
        return all_duplicates
    
    def load_data(self, data_type='clean'):
        """
        Load data from both sources
        
        Args:
            data_type: 'clean' or 'features'
        
        Returns:
            Tuple of (df_topcv, df_vnw) or (None, None) if error
        """
        path_key = 'clean_path' if data_type == 'clean' else 'features_path'
        
        # Load TopCV
        topcv_path = SOURCES['topcv'][path_key]
        print(f"\n📂 Loading TopCV {data_type} data...")
        print(f"   File: {topcv_path}")
        
        if not topcv_path.exists():
            print(f"❌ Not found: {topcv_path}")
            return None, None
        
        df_topcv = pd.read_csv(topcv_path)
        df_topcv['source'] = 'topcv'
        self.stats['topcv_jobs'] = len(df_topcv)
        print(f"✅ Loaded: {len(df_topcv):,} jobs")
        
        # Load VietnamWorks
        vnw_path = SOURCES['vietnamworks'][path_key]
        print(f"\n📂 Loading VietnamWorks {data_type} data...")
        print(f"   File: {vnw_path}")
        
        if not vnw_path.exists():
            print(f"⚠️  Not found: {vnw_path}")
            print(f"   Will use TopCV only")
            return df_topcv, pd.DataFrame()
        
        df_vnw = pd.read_csv(vnw_path)
        df_vnw['source'] = 'vietnamworks'
        self.stats['vnw_jobs'] = len(df_vnw)
        print(f"✅ Loaded: {len(df_vnw):,} jobs")
        
        return df_topcv, df_vnw
    
    def merge_dataframes(self, df_topcv, df_vnw):
        """
        Merge two dataframes after deduplication
        
        Args:
            df_topcv: TopCV dataframe
            df_vnw: VietnamWorks dataframe
        
        Returns:
            Merged dataframe
        """
        # If VietnamWorks is empty, return TopCV only
        if len(df_vnw) == 0:
            print("\n⚠️  No VietnamWorks data to merge")
            return df_topcv
        
        # Detect duplicates
        dup_indices = self.detect_duplicates(df_topcv, df_vnw)
        
        # Remove duplicates from VietnamWorks
        print(f"\n🗑️  Removing duplicates from VietnamWorks...")
        df_vnw_dedup = df_vnw[~df_vnw.index.isin(dup_indices)].copy()
        print(f"   Before: {len(df_vnw):,} jobs")
        print(f"   After:  {len(df_vnw_dedup):,} jobs")
        print(f"   Removed: {len(dup_indices):,} duplicates")
        
        # Merge
        print(f"\n🔗 Merging datasets...")
        df_merged = pd.concat([df_topcv, df_vnw_dedup], ignore_index=True)
        
        # Sort by source and title
        df_merged = df_merged.sort_values(['source', 'tieu_de']).reset_index(drop=True)
        
        self.stats['final_jobs'] = len(df_merged)
        
        return df_merged
    
    def save_merged_data(self, df, output_path, data_type='clean'):
        """
        Save merged data
        
        Args:
            df: Merged dataframe
            output_path: Output file path
            data_type: 'clean' or 'features'
        """
        print(f"\n💾 Saving merged {data_type} data...")
        print(f"   File: {output_path}")
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"✅ Saved: {len(df):,} jobs, {len(df.columns)} columns")
        print(f"   Size: {file_size_mb:.2f} MB")
    
    def print_statistics(self, df):
        """Print merge statistics"""
        print("\n" + "="*60)
        print("📊 MERGE STATISTICS")
        print("="*60)
        
        print(f"\n📋 Input:")
        print(f"   TopCV jobs:        {self.stats['topcv_jobs']:,}")
        print(f"   VietnamWorks jobs: {self.stats['vnw_jobs']:,}")
        print(f"   Total input:       {self.stats['topcv_jobs'] + self.stats['vnw_jobs']:,}")
        
        print(f"\n🔍 Deduplication:")
        print(f"   Strategy:          {self.dedup_strategy}")
        print(f"   Similarity:        {self.similarity_threshold}")
        print(f"   Exact duplicates:  {self.stats['exact_duplicates']:,}")
        print(f"   Fuzzy duplicates:  {self.stats['fuzzy_duplicates']:,}")
        print(f"   Total duplicates:  {self.stats['total_duplicates']:,}")
        
        print(f"\n📊 Output:")
        print(f"   Final jobs:        {self.stats['final_jobs']:,}")
        
        # Distribution by source
        source_dist = df['source'].value_counts()
        print(f"\n🗂️  Distribution by source:")
        for source, count in source_dist.items():
            pct = count / len(df) * 100
            print(f"   {source:15s}: {count:5,} ({pct:5.1f}%)")
        
        # Additional stats if features available
        if 'job_group' in df.columns:
            print(f"\n🏷️  Job groups:")
            job_groups = df['job_group'].value_counts()
            for group, count in job_groups.head(5).items():
                pct = count / len(df) * 100
                print(f"   {group:15s}: {count:5,} ({pct:5.1f}%)")
        
        if 'n_skills' in df.columns:
            print(f"\n🔧 Skills:")
            print(f"   Avg skills/job:    {df['n_skills'].mean():.2f}")
            print(f"   Jobs with skills:  {(df['n_skills'] > 0).sum():,} ({(df['n_skills'] > 0).sum()/len(df)*100:.1f}%)")
        
        if 'salary_avg' in df.columns:
            n_salary = df['salary_avg'].notna().sum()
            print(f"\n💰 Salary:")
            print(f"   Jobs with salary:  {n_salary:,} ({n_salary/len(df)*100:.1f}%)")
            if n_salary > 0:
                print(f"   Avg salary:        {df['salary_avg'].mean():.1f}M VND")
    
    def merge_clean_data(self):
        """Merge cleaned data"""
        print("\n" + "="*60)
        print("🔄 MERGING CLEAN DATA")
        print("="*60)
        
        # Load
        df_topcv, df_vnw = self.load_data('clean')
        
        if df_topcv is None:
            print("❌ Failed to load data")
            return False
        
        # Merge
        df_merged = self.merge_dataframes(df_topcv, df_vnw)
        
        # Save
        self.save_merged_data(df_merged, MERGED_CLEAN_PATH, 'clean')
        
        # Statistics
        self.print_statistics(df_merged)
        
        return True
    
    def merge_features(self):
        """Merge feature data"""
        print("\n" + "="*60)
        print("🔄 MERGING FEATURE DATA")
        print("="*60)
        
        # Load
        df_topcv, df_vnw = self.load_data('features')
        
        if df_topcv is None:
            print("❌ Failed to load data")
            return False
        
        # Merge
        df_merged = self.merge_dataframes(df_topcv, df_vnw)
        
        # Save
        self.save_merged_data(df_merged, MERGED_FEATURES_PATH, 'features')
        
        # Statistics
        self.print_statistics(df_merged)
        
        return True
    
    def run(self):
        """Run complete merge pipeline"""
        print("\n" + "="*60)
        print("🔀 DATA MERGE PIPELINE")
        print("="*60)
        
        # Merge clean data
        success_clean = self.merge_clean_data()
        
        if not success_clean:
            print("\n❌ Clean data merge failed")
            return False
        
        # Merge features
        success_features = self.merge_features()
        
        if not success_features:
            print("\n❌ Feature data merge failed")
            return False
        
        print("\n" + "="*60)
        print("✅ MERGE COMPLETED")
        print("="*60)
        
        print(f"\n📁 Output files:")
        print(f"   Clean:    {MERGED_CLEAN_PATH}")
        print(f"   Features: {MERGED_FEATURES_PATH}")
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Merge TopCV and VietnamWorks data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (fuzzy matching, 0.85 threshold)
  python3 src/05_merge_sources.py
  
  # Exact matching only
  python3 src/05_merge_sources.py --dedup-strategy exact
  
  # Both exact and fuzzy matching
  python3 src/05_merge_sources.py --dedup-strategy both
  
  # Fuzzy matching with higher threshold
  python3 src/05_merge_sources.py --similarity-threshold 0.9
        """
    )
    
    parser.add_argument(
        '--dedup-strategy',
        type=str,
        choices=['exact', 'fuzzy', 'both'],
        default='fuzzy',
        help='Deduplication strategy (default: fuzzy)'
    )
    
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.85,
        help='Title similarity threshold for fuzzy matching (default: 0.85)'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.similarity_threshold <= 1.0:
        print("❌ Error: similarity-threshold must be between 0.0 and 1.0")
        return 1
    
    # Create merger
    merger = DataMerger(
        similarity_threshold=args.similarity_threshold,
        dedup_strategy=args.dedup_strategy
    )
    
    # Run merge
    success = merger.run()
    
    if not success:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())