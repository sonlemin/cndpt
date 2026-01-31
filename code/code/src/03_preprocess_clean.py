#!/usr/bin/env python3
# src/03_preprocess_clean.py
"""
Data Cleaning & Preprocessing

Cleans and preprocesses job posting data from different sources.
Supports: TopCV, VietnamWorks, or both.

Usage:
    python3 src/03_preprocess_clean.py --source topcv
    python3 src/03_preprocess_clean.py --source vietnamworks
    python3 src/03_preprocess_clean.py --source both
"""

import pandas as pd
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities
try:
    from utils import clean_text_vn, clean_url
except ImportError:
    print("⚠️  Warning: utils module not found. Using basic cleaning functions.")
    
    def clean_text_vn(text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        text = str(text).strip()
        # Remove multiple spaces
        import re
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def clean_url(url):
        """Basic URL cleaning"""
        if pd.isna(url):
            return ""
        url = str(url).strip()
        return url


class JobDataCleaner:
    """Job posting data cleaner"""
    
    def __init__(self, source='topcv'):
        """
        Initialize cleaner
        
        Args:
            source: 'topcv', 'vietnamworks', or 'both'
        """
        self.source = source.lower()
        self.stats = {
            'initial_rows': 0,
            'after_dropna': 0,
            'after_dedup': 0,
            'after_length_filter': 0,
            'final_rows': 0,
            'duplicates_removed': 0,
            'invalid_removed': 0,
        }
        
        # Paths
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.min_raw_length = 800       # Min raw content length
        self.min_clean_length = 200     # Min cleaned content length
        self.required_columns = ['tieu_de', 'link', 'noi_dung']
    
    def get_paths(self, source):
        """Get file paths for a source"""
        if source == 'topcv':
            raw_path = self.raw_dir / "topcv_it_detail.csv"
            clean_path = self.processed_dir / "topcv_it_clean.csv"
        elif source == 'vietnamworks':
            raw_path = self.raw_dir / "vietnamworks_it_detail.csv"
            clean_path = self.processed_dir / "vietnamworks_it_clean.csv"
        else:
            raise ValueError(f"Invalid source: {source}")
        
        return raw_path, clean_path
    
    def load_data(self, source):
        """Load raw data"""
        raw_path, _ = self.get_paths(source)
        
        print(f"\n📂 Loading {source.upper()} data...")
        print(f"   File: {raw_path}")
        
        if not raw_path.exists():
            print(f"❌ Error: File not found: {raw_path}")
            return None
        
        try:
            df = pd.read_csv(raw_path)
            print(f"✅ Loaded {len(df)} rows")
            
            # Check required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                print(f"❌ Error: Missing required columns: {missing_cols}")
                return None
            
            return df
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def validate_data(self, df, stage="initial"):
        """Validate data quality"""
        print(f"\n📊 Data Validation ({stage}):")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        # Check for nulls
        null_counts = df[self.required_columns].isnull().sum()
        if null_counts.sum() > 0:
            print(f"\n   Null values:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"      {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Check for duplicates
        if 'link' in df.columns:
            dup_count = df['link'].duplicated().sum()
            if dup_count > 0:
                print(f"\n   Duplicates (by link): {dup_count} ({dup_count/len(df)*100:.1f}%)")
        
        # Content length stats
        if 'noi_dung' in df.columns:
            lengths = df['noi_dung'].astype(str).str.len()
            print(f"\n   Content length:")
            print(f"      Mean: {lengths.mean():.0f} chars")
            print(f"      Median: {lengths.median():.0f} chars")
            print(f"      Min: {lengths.min():.0f} chars")
            print(f"      Max: {lengths.max():.0f} chars")
    
    def clean_data(self, df):
        """Clean the dataframe"""
        print(f"\n🧹 Cleaning data...")
        
        self.stats['initial_rows'] = len(df)
        initial_rows = len(df)
        
        # Step 1: Drop rows with missing critical columns
        print(f"\n   Step 1: Removing rows with missing values...")
        df = df.dropna(subset=self.required_columns).copy()
        self.stats['after_dropna'] = len(df)
        removed = initial_rows - len(df)
        if removed > 0:
            print(f"      Removed {removed} rows ({removed/initial_rows*100:.1f}%)")
        
        # Step 2: Clean URLs
        print(f"\n   Step 2: Cleaning URLs...")
        df['link'] = df['link'].astype(str).apply(clean_url)
        df['link'] = df['link'].replace({'nan': pd.NA, '': pd.NA})
        
        # Remove rows with invalid URLs
        before = len(df)
        df = df.dropna(subset=['link'])
        removed = before - len(df)
        if removed > 0:
            print(f"      Removed {removed} rows with invalid URLs")
        
        # Step 3: Remove duplicates
        print(f"\n   Step 3: Removing duplicates...")
        before = len(df)
        df = df.drop_duplicates(subset=['link'])
        self.stats['after_dedup'] = len(df)
        self.stats['duplicates_removed'] = before - len(df)
        
        if self.stats['duplicates_removed'] > 0:
            print(f"      Removed {self.stats['duplicates_removed']} duplicates ({self.stats['duplicates_removed']/before*100:.1f}%)")
        
        # Step 4: Clean text fields
        print(f"\n   Step 4: Cleaning text fields...")
        df['tieu_de_clean'] = df['tieu_de'].astype(str).apply(clean_text_vn)
        df['noi_dung_clean'] = df['noi_dung'].astype(str).apply(clean_text_vn)
        
        # Step 5: Filter by content length
        print(f"\n   Step 5: Filtering by content length...")
        print(f"      Min raw length: {self.min_raw_length} chars")
        print(f"      Min clean length: {self.min_clean_length} chars")
        
        raw_len = df['noi_dung'].astype(str).str.len()
        clean_len = df['noi_dung_clean'].astype(str).str.len()
        
        # Keep rows that meet either threshold
        mask = (raw_len >= self.min_raw_length) | (clean_len >= self.min_clean_length)
        
        before = len(df)
        df = df[mask].copy()
        self.stats['after_length_filter'] = len(df)
        removed = before - len(df)
        
        if removed > 0:
            print(f"      Removed {removed} rows ({removed/before*100:.1f}%)")
        
        # Add metadata
        df['source'] = self.source if self.source != 'both' else 'unknown'
        
        self.stats['final_rows'] = len(df)
        self.stats['invalid_removed'] = initial_rows - len(df)
        
        return df
    
    def save_data(self, df, source):
        """Save cleaned data"""
        _, clean_path = self.get_paths(source)
        
        print(f"\n💾 Saving cleaned data...")
        print(f"   File: {clean_path}")
        
        df.to_csv(clean_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ Saved {len(df)} rows")
        
        # Save column info
        print(f"\n   Columns ({len(df.columns)}):")
        for col in df.columns:
            print(f"      - {col}")
    
    def print_statistics(self):
        """Print cleaning statistics"""
        print(f"\n" + "="*60)
        print(f"📊 CLEANING STATISTICS")
        print("="*60)
        
        print(f"\nRows:")
        print(f"   Initial:             {self.stats['initial_rows']:,}")
        print(f"   After dropna:        {self.stats['after_dropna']:,}")
        print(f"   After dedup:         {self.stats['after_dedup']:,}")
        print(f"   After length filter: {self.stats['after_length_filter']:,}")
        print(f"   Final:               {self.stats['final_rows']:,}")
        
        print(f"\nRemoved:")
        print(f"   Duplicates:          {self.stats['duplicates_removed']:,}")
        print(f"   Invalid/short:       {self.stats['invalid_removed']:,}")
        
        if self.stats['initial_rows'] > 0:
            retention = self.stats['final_rows'] / self.stats['initial_rows'] * 100
            print(f"\nRetention rate:        {retention:.1f}%")
    
    def process(self, source):
        """Process a single source"""
        print("\n" + "="*60)
        print(f"🔄 PROCESSING {source.upper()}")
        print("="*60)
        
        # Load
        df = self.load_data(source)
        if df is None:
            return False
        
        # Validate before
        self.validate_data(df, stage="before cleaning")
        
        # Clean
        df_clean = self.clean_data(df)
        
        # Validate after
        self.validate_data(df_clean, stage="after cleaning")
        
        # Save
        self.save_data(df_clean, source)
        
        # Statistics
        self.print_statistics()
        
        return True
    
    def run(self):
        """Run cleaning for specified source(s)"""
        print("\n" + "="*60)
        print("🧹 DATA CLEANING & PREPROCESSING")
        print("="*60)
        print(f"\nSource: {self.source.upper()}")
        
        if self.source == 'both':
            # Process both sources
            sources = ['topcv', 'vietnamworks']
            
            for src in sources:
                success = self.process(src)
                if not success:
                    print(f"\n⚠️  Skipping {src.upper()} (data not found or error)")
        
        else:
            # Process single source
            success = self.process(self.source)
            if not success:
                print(f"\n❌ Failed to process {self.source.upper()}")
                return False
        
        print("\n" + "="*60)
        print("✅ CLEANING COMPLETED")
        print("="*60)
        
        return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Clean and preprocess job posting data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 src/03_preprocess_clean.py --source topcv
  python3 src/03_preprocess_clean.py --source vietnamworks
  python3 src/03_preprocess_clean.py --source both
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        choices=['topcv', 'vietnamworks', 'both'],
        default='topcv',
        help='Data source to clean (default: topcv)'
    )
    
    parser.add_argument(
        '--min-raw-length',
        type=int,
        default=800,
        help='Minimum raw content length (default: 800)'
    )
    
    parser.add_argument(
        '--min-clean-length',
        type=int,
        default=200,
        help='Minimum cleaned content length (default: 200)'
    )
    
    args = parser.parse_args()
    
    # Create cleaner
    cleaner = JobDataCleaner(source=args.source)
    
    # Update thresholds if provided
    cleaner.min_raw_length = args.min_raw_length
    cleaner.min_clean_length = args.min_clean_length
    
    # Run cleaning
    success = cleaner.run()
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()