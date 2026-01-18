#!/usr/bin/env python3
"""
debug_java_php.py

Script debug chi tiết để tìm tại sao java/php luôn match

USAGE:
    cd ~/workspace/github.com/sonlemin/cndpt/code/code
    python src/debug_java_php.py
"""

import pandas as pd
import re

def debug_patterns():
    """Debug java/php patterns"""
    print("="*80)
    print("🐛 DEBUG JAVA/PHP PATTERNS")
    print("="*80)
    
    # Import config
    print("\n📦 Importing config...")
    try:
        from config_v2 import SKILL_PATTERNS
        config_name = "config_v2"
    except ImportError:
        try:
            from config_improved import SKILL_PATTERNS
            config_name = "config_improved"
        except ImportError:
            from config import SKILL_PATTERNS
            config_name = "config"
    
    print(f"✅ Using: {config_name}.py")
    
    # Check patterns
    print(f"\n" + "="*80)
    print("🔍 CHECK PATTERNS")
    print("="*80)
    
    java_pattern = SKILL_PATTERNS.get('java', None)
    php_pattern = SKILL_PATTERNS.get('php', None)
    
    print(f"\nJava pattern: {java_pattern}")
    print(f"PHP pattern:  {php_pattern}")
    
    # Test patterns on sample text
    print(f"\n" + "="*80)
    print("🧪 TEST PATTERNS")
    print("="*80)
    
    test_cases = [
        ("Mobile Developer", "mobile app ios swift"),
        ("QA Tester", "testing automation selenium"),
        ("UI Designer", "figma sketch photoshop design"),
        ("Java Developer", "java spring boot backend"),
        ("PHP Developer", "php laravel mysql backend"),
        ("Empty", ""),
    ]
    
    print("\nTesting java pattern:")
    print("-"*80)
    for title, text in test_cases:
        match = re.search(java_pattern, text, re.IGNORECASE) if java_pattern else None
        status = "✅ MATCH" if match else "❌ NO MATCH"
        print(f"{title:20s}: {status}")
        if match:
            print(f"  → Matched: '{match.group()}'")
    
    print("\nTesting php pattern:")
    print("-"*80)
    for title, text in test_cases:
        match = re.search(php_pattern, text, re.IGNORECASE) if php_pattern else None
        status = "✅ MATCH" if match else "❌ NO MATCH"
        print(f"{title:20s}: {status}")
        if match:
            print(f"  → Matched: '{match.group()}'")
    
    # Test on real data
    print(f"\n" + "="*80)
    print("📊 TEST ON REAL DATA")
    print("="*80)
    
    try:
        df = pd.read_csv("data/processed/topcv_it_clean.csv")
    except FileNotFoundError:
        print("❌ Clean CSV not found")
        return
    
    print(f"\nTesting first 5 jobs:")
    print("-"*80)
    
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        
        title = row.get('tieu_de', 'N/A')[:60]
        
        text = ""
        if "tieu_de_clean" in df.columns and pd.notna(row.get("tieu_de_clean")):
            text += str(row["tieu_de_clean"]).lower() + " "
        if "mo_ta_clean" in df.columns and pd.notna(row.get("mo_ta_clean")):
            text += str(row["mo_ta_clean"]).lower() + " "
        
        print(f"\n{idx+1}. {title}")
        print(f"   Text length: {len(text)} chars")
        print(f"   Text sample: {text[:100]}...")
        
        # Test java
        java_match = re.search(java_pattern, text, re.IGNORECASE) if java_pattern else None
        if java_match:
            print(f"   ✅ Java MATCH: '{java_match.group()}'")
            # Show context
            start = max(0, java_match.start() - 30)
            end = min(len(text), java_match.end() + 30)
            context = text[start:end]
            print(f"      Context: ...{context}...")
        else:
            print(f"   ❌ Java NO MATCH")
        
        # Test php
        php_match = re.search(php_pattern, text, re.IGNORECASE) if php_pattern else None
        if php_match:
            print(f"   ✅ PHP MATCH: '{php_match.group()}'")
            # Show context
            start = max(0, php_match.start() - 30)
            end = min(len(text), php_match.end() + 30)
            context = text[start:end]
            print(f"      Context: ...{context}...")
        else:
            print(f"   ❌ PHP NO MATCH")
    
    # Diagnosis
    print(f"\n" + "="*80)
    print("💡 CHẨN ĐOÁN")
    print("="*80)
    
    # Check if patterns are too broad
    if java_pattern in [r".*", r".+", r"", None]:
        print(f"\n❌ BUG FOUND: Java pattern is too broad or empty!")
        print(f"   Pattern: {java_pattern}")
    
    if php_pattern in [r".*", r".+", r"", None]:
        print(f"\n❌ BUG FOUND: PHP pattern is too broad or empty!")
        print(f"   Pattern: {php_pattern}")
    
    # Check if patterns always match
    empty_text = ""
    java_empty = re.search(java_pattern, empty_text, re.IGNORECASE) if java_pattern else None
    php_empty = re.search(php_pattern, empty_text, re.IGNORECASE) if php_pattern else None
    
    if java_empty:
        print(f"\n❌ CRITICAL BUG: Java pattern matches EMPTY string!")
        print(f"   Pattern: {java_pattern}")
    
    if php_empty:
        print(f"\n❌ CRITICAL BUG: PHP pattern matches EMPTY string!")
        print(f"   Pattern: {php_pattern}")
    
    print(f"\n" + "="*80)
    print("📝 RECOMMENDED FIX")
    print("="*80)
    
    print("\nCurrent patterns:")
    print(f"  java: {java_pattern}")
    print(f"  php:  {php_pattern}")
    
    print("\nRecommended patterns:")
    print(f"  java: r\"\\bjava\\b(?!\\s*script)\"")
    print(f"  php:  r\"\\bphp\\b\"")
    
    print("\nThese patterns should:")
    print("  ✅ Match 'java' as whole word")
    print("  ✅ NOT match 'javascript'")
    print("  ✅ NOT match empty string")
    print("  ✅ Be case-insensitive")

if __name__ == "__main__":
    debug_patterns()