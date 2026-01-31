#!/usr/bin/env python3
# src/04b_add_regional_features.py
"""
Add Regional Features - Thêm features về vùng miền

Script này extract regional features từ data hiện có:
- Detect location từ job descriptions, titles, và các fields khác
- Map locations to regions (North/South/Central)
- Detect remote work flags
- Add company size estimates

Input:  data/processed/topcv_it_features.csv
Output: data/processed/topcv_it_features_with_regions.csv
"""

import pandas as pd
import re
from pathlib import Path

# Paths
INPUT_PATH = "data/processed/topcv_it_features.csv"
OUTPUT_PATH = "data/processed/topcv_it_features_with_regions.csv"

# Location mappings
REGION_MAPPING = {
    'North': [
        'hà nội', 'ha noi', 'hanoi', 'hn',
        'hải phòng', 'hai phong', 'haiphong',
        'quảng ninh', 'quang ninh',
        'thái nguyên', 'thai nguyen',
        'bắc ninh', 'bac ninh',
        'bắc giang', 'bac giang',
        'vĩnh phúc', 'vinh phuc',
        'hưng yên', 'hung yen',
        'hà nam', 'ha nam',
        'nam định', 'nam dinh',
        'thái bình', 'thai binh',
        'ninh bình', 'ninh binh'
    ],
    'Central': [
        'đà nẵng', 'da nang', 'danang', 'đn',
        'huế', 'hue',
        'quảng nam', 'quang nam',
        'quảng ngãi', 'quang ngai',
        'bình định', 'binh dinh',
        'phú yên', 'phu yen',
        'khánh hòa', 'khanh hoa', 'nha trang',
        'đắk lắk', 'dak lak', 'buon ma thuot'
    ],
    'South': [
        'tp.hcm', 'tp hcm', 'hồ chí minh', 'ho chi minh', 'hcm', 'saigon', 'sài gòn',
        'bình dương', 'binh duong', 'thủ dầu một', 'thu dau mot',
        'đồng nai', 'dong nai', 'biên hòa', 'bien hoa',
        'bà rịa', 'ba ria', 'vũng tàu', 'vung tau',
        'long an', 'tân an', 'tan an',
        'tiền giang', 'tien giang', 'mỹ tho', 'my tho',
        'bến tre', 'ben tre',
        'vĩnh long', 'vinh long',
        'cần thơ', 'can tho',
        'an giang', 'châu đốc', 'chau doc',
        'kiên giang', 'kien giang', 'rạch giá', 'rach gia',
        'cà mau', 'ca mau',
        'sóc trăng', 'soc trang',
        'bạc liêu', 'bac lieu',
        'hậu giang', 'hau giang'
    ]
}

# Remote work keywords
REMOTE_KEYWORDS = [
    'remote', 'từ xa', 'tu xa', 'wfh', 'work from home',
    'làm việc tại nhà', 'lam viec tai nha',
    'online', 'telecommute'
]

# Hybrid work keywords
HYBRID_KEYWORDS = [
    'hybrid', 'lai', 'kết hợp', 'ket hop',
    'part-time office', 'flexible',
    'linh hoạt', 'linh hoat'
]


def detect_location(text):
    """Detect location from text"""
    if pd.isna(text):
        return None
    
    text = str(text).lower()
    
    # Check each region
    for region, cities in REGION_MAPPING.items():
        for city in cities:
            if city in text:
                return region, city
    
    return None, None


def detect_primary_city(text):
    """Extract primary city/province name"""
    if pd.isna(text):
        return None
    
    text = str(text).lower()
    
    # Priority order: HCM > Hanoi > Danang > Others
    priority_cities = [
        ('TP.HCM', ['tp.hcm', 'tp hcm', 'hồ chí minh', 'ho chi minh', 'hcm', 'saigon']),
        ('Hà Nội', ['hà nội', 'ha noi', 'hanoi', 'hn']),
        ('Đà Nẵng', ['đà nẵng', 'da nang', 'danang']),
        ('Hải Phòng', ['hải phòng', 'hai phong']),
        ('Cần Thơ', ['cần thơ', 'can tho']),
        ('Bình Dương', ['bình dương', 'binh duong']),
        ('Đồng Nai', ['đồng nai', 'dong nai']),
    ]
    
    for city_name, keywords in priority_cities:
        for keyword in keywords:
            if keyword in text:
                return city_name
    
    return 'Khác'


def detect_remote_work(text):
    """Detect if job allows remote work"""
    if pd.isna(text):
        return False
    
    text = str(text).lower()
    
    for keyword in REMOTE_KEYWORDS:
        if keyword in text:
            return True
    
    return False


def detect_hybrid_work(text):
    """Detect if job is hybrid"""
    if pd.isna(text):
        return False
    
    text = str(text).lower()
    
    for keyword in HYBRID_KEYWORDS:
        if keyword in text:
            return True
    
    return False


def estimate_company_size(text):
    """Estimate company size from description"""
    if pd.isna(text):
        return 'Unknown'
    
    text = str(text).lower()
    
    # Large company indicators
    large_indicators = [
        'tập đoàn', 'tap doan', 'corporation', 'group',
        '1000+', '500+', 'multinational', 'đa quốc gia',
        'fortune 500', 'listed company', 'công ty niêm yết'
    ]
    
    # Startup indicators
    startup_indicators = [
        'startup', 'khởi nghiệp', 'khoi nghiep',
        'early stage', 'seed', 'series a',
        'disruptive', 'innovative'
    ]
    
    # SME indicators
    sme_indicators = [
        'sme', 'small', 'medium',
        'vừa và nhỏ', 'vua va nho',
        '10-50', '50-100'
    ]
    
    for indicator in large_indicators:
        if indicator in text:
            return 'Large (500+)'
    
    for indicator in startup_indicators:
        if indicator in text:
            return 'Startup (<50)'
    
    for indicator in sme_indicators:
        if indicator in text:
            return 'SME (50-500)'
    
    return 'Medium (100-500)'  # Default


def add_regional_features(df):
    """Add regional features to dataframe"""
    print("\n" + "="*60)
    print("🗺️  ADDING REGIONAL FEATURES")
    print("="*60)
    
    # Combine text fields for location detection
    df['combined_text'] = df.apply(
        lambda row: ' '.join([
            str(row.get('tieu_de', '')),
            str(row.get('mo_ta', '')),
            str(row.get('link', '')),
        ]).lower(),
        axis=1
    )
    
    # 1. Detect region and city
    print("\n1. Detecting regions...")
    location_data = df['combined_text'].apply(detect_location)
    df['region'] = location_data.apply(lambda x: x[0] if x else None)
    df['detected_city'] = location_data.apply(lambda x: x[1] if x else None)
    
    # Get primary location
    df['primary_location'] = df['combined_text'].apply(detect_primary_city)
    
    region_dist = df['region'].value_counts()
    print(f"   ✅ Regions detected:")
    for region, count in region_dist.items():
        print(f"      {region:10s}: {count:3d} jobs ({count/len(df)*100:5.1f}%)")
    
    unknown_count = df['region'].isna().sum()
    if unknown_count > 0:
        print(f"      {'Unknown':10s}: {unknown_count:3d} jobs ({unknown_count/len(df)*100:5.1f}%)")
    
    # 2. Detect remote work
    print("\n2. Detecting remote work...")
    df['is_remote'] = df['combined_text'].apply(detect_remote_work)
    remote_count = df['is_remote'].sum()
    print(f"   ✅ Remote jobs: {remote_count} ({remote_count/len(df)*100:.1f}%)")
    
    # 3. Detect hybrid work
    print("\n3. Detecting hybrid work...")
    df['is_hybrid'] = df['combined_text'].apply(detect_hybrid_work)
    hybrid_count = df['is_hybrid'].sum()
    print(f"   ✅ Hybrid jobs: {hybrid_count} ({hybrid_count/len(df)*100:.1f}%)")
    
    # 4. Estimate company size
    print("\n4. Estimating company sizes...")
    df['company_size'] = df['combined_text'].apply(estimate_company_size)
    size_dist = df['company_size'].value_counts()
    print(f"   ✅ Company sizes:")
    for size, count in size_dist.items():
        print(f"      {size:20s}: {count:3d} jobs ({count/len(df)*100:5.1f}%)")
    
    # Drop temporary column
    df = df.drop('combined_text', axis=1)
    
    return df


def main():
    """Main function"""
    print("\n" + "="*60)
    print("🏭 ADD REGIONAL FEATURES TO IT JOBS DATA")
    print("="*60)
    
    # Load data
    print(f"\n📂 Loading data from: {INPUT_PATH}")
    
    if not Path(INPUT_PATH).exists():
        print(f"❌ File not found: {INPUT_PATH}")
        print(f"\n   Please run feature extraction first:")
        print(f"   python3 src/04_extract_features.py --source topcv")
        exit(1)
    
    df = pd.read_csv(INPUT_PATH)
    print(f"✅ Loaded {len(df)} jobs")
    print(f"   Columns: {len(df.columns)}")
    
    # Add regional features
    df = add_regional_features(df)
    
    # Save
    print(f"\n💾 Saving data with regional features...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved to: {OUTPUT_PATH}")
    print(f"   Total columns: {len(df.columns)}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ REGIONAL FEATURES ADDED SUCCESSFULLY")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   Input:  {INPUT_PATH}")
    print(f"   Output: {OUTPUT_PATH}")
    print(f"   Jobs:   {len(df)}")
    
    new_cols = ['region', 'primary_location', 'is_remote', 'is_hybrid', 'company_size']
    print(f"\n   New features added:")
    for col in new_cols:
        if col in df.columns:
            print(f"      ✅ {col}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Update scripts 07, 08 to use: {OUTPUT_PATH}")
    print(f"   2. Or copy to original path:")
    print(f"      cp {OUTPUT_PATH} {INPUT_PATH}")
    print(f"   3. Run regional analysis:")
    print(f"      python3 src/08_regional_analysis.py")
    

if __name__ == "__main__":
    main()