#!/usr/bin/env python3
# src/04_extract_features.py
"""
Feature Extraction from Job Postings

Extracts comprehensive features from cleaned job data.

Usage:
    python3 src/04_extract_features.py --source topcv
    python3 src/04_extract_features.py --source vietnamworks  
    python3 src/04_extract_features.py --source both
"""

import pandas as pd
import re
import argparse
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import config
try:
    from config import SOURCES
except ImportError:
    print("⚠️  Config not found, using defaults")
    SOURCES = {
        'topcv': {
            'clean_path': Path('data/processed/topcv_it_clean.csv'),
            'features_path': Path('data/processed/topcv_it_features.csv'),
        },
        'vietnamworks': {
            'clean_path': Path('data/processed/vietnamworks_it_clean.csv'),
            'features_path': Path('data/processed/vietnamworks_it_features.csv'),
        }
    }


# Job group patterns
JOB_GROUP_PATTERNS = [
    ('backend', r'\b(backend|back-end|back end|api|server|microservice)\b'),
    ('frontend', r'\b(frontend|front-end|front end|ui|web developer)\b'),
    ('fullstack', r'\b(fullstack|full-stack|full stack)\b'),
    ('mobile', r'\b(mobile|ios|android|react native|flutter)\b'),
    ('devops', r'\b(devops|dev-ops|infrastructure|sre)\b'),
    ('data', r'\b(data\s+(?:scientist|engineer|analyst)|machine learning|ml|ai)\b'),
    ('qa', r'\b(qa|qc|test|quality assurance)\b'),
    ('security', r'\b(security|bảo mật|cybersecurity)\b'),
    ('pm', r'\b(project manager|pm|scrum master|product owner)\b'),
    ('ba', r'\b(business analyst|ba|system analyst)\b'),
]

# Skills to detect (context-aware)
SKILLS = [
    'JavaScript', 'Python', 'Java', 'C#', 'PHP', 'TypeScript', 'Go', 'Ruby', 'C++',
    'React', 'Angular', 'Vue', 'Next.js', 'Node.js', 'Django', 'Spring', 'Laravel', 'Flask',
    'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle',
    'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'CI/CD',
    'Git', 'Jira', 'Agile', 'Scrum',
    'SQL', 'HTML', 'CSS',
]


class FeatureExtractor:
    """Extract features from job postings"""
    
    def __init__(self, source='topcv'):
        self.source = source.lower()
        self.processed_dir = Path("data/processed")
        
        # Build skill patterns
        self.skill_patterns = self._build_skill_patterns()
    
    def _build_skill_patterns(self):
        """Build context-aware skill patterns to reduce false positives"""
        patterns = {}
        
        # Context keywords
        context = r'(?:developer|engineer|lập trình|experience|kinh nghiệm|skill|know|biết|yêu cầu)'
        
        # Languages (require context in same text)
        patterns['JavaScript'] = r'\b(?:javascript|js)\b'
        patterns['Python'] = r'\bpython\b'
        patterns['Java'] = r'\bjava\b(?!script)'
        patterns['C#'] = r'\bc#\b'
        patterns['PHP'] = r'\bphp\b'
        patterns['TypeScript'] = r'\b(?:typescript|ts)\b'
        patterns['Go'] = r'\b(?:golang|go\s+lang)\b'
        patterns['Ruby'] = r'\bruby\b'
        patterns['C++'] = r'\bc\+\+\b'
        
        # Frameworks
        patterns['React'] = r'\breact(?:js)?\b'
        patterns['Angular'] = r'\bangular\b'
        patterns['Vue'] = r'\bvue(?:\.js)?\b'
        patterns['Next.js'] = r'\bnext(?:\.js)?\b'
        patterns['Node.js'] = r'\bnode(?:\.js)?\b'
        patterns['Django'] = r'\bdjango\b'
        patterns['Spring'] = r'\bspring(?:\s+boot)?\b'
        patterns['Laravel'] = r'\blaravel\b'
        patterns['Flask'] = r'\bflask\b'
        
        # Databases
        patterns['MySQL'] = r'\bmysql\b'
        patterns['PostgreSQL'] = r'\b(?:postgresql|postgres)\b'
        patterns['MongoDB'] = r'\bmongodb\b'
        patterns['Redis'] = r'\bredis\b'
        patterns['Oracle'] = r'\boracle\b'
        
        # Cloud & DevOps
        patterns['AWS'] = r'\b(?:aws|amazon web services)\b'
        patterns['Azure'] = r'\b(?:azure|microsoft azure)\b'
        patterns['GCP'] = r'\b(?:gcp|google cloud)\b'
        patterns['Docker'] = r'\bdocker\b'
        patterns['Kubernetes'] = r'\b(?:kubernetes|k8s)\b'
        patterns['CI/CD'] = r'\b(?:ci/cd|jenkins|gitlab ci)\b'
        
        # Tools
        patterns['Git'] = r'\bgit\b(?!hub|lab)'
        patterns['Jira'] = r'\bjira\b'
        patterns['Agile'] = r'\bagile\b'
        patterns['Scrum'] = r'\bscrum\b'
        
        # General
        patterns['SQL'] = r'\bsql\b'
        patterns['HTML'] = r'\bhtml5?\b'
        patterns['CSS'] = r'\bcss3?\b'
        
        return patterns
    
    def get_paths(self, source):
        """Get file paths"""
        if source not in SOURCES:
            raise ValueError(f"Unknown source: {source}")
        return SOURCES[source]['clean_path'], SOURCES[source]['features_path']
    
    def load_data(self, source):
        """Load cleaned data"""
        clean_path, _ = self.get_paths(source)
        
        print(f"\n📂 Loading {source.upper()}...")
        print(f"   File: {clean_path}")
        
        if not clean_path.exists():
            print(f"❌ Not found: {clean_path}")
            print(f"   Run: python3 src/03_preprocess_clean.py --source {source}")
            return None
        
        try:
            df = pd.read_csv(clean_path)
            print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Check required columns
            required = ['tieu_de_clean', 'noi_dung_clean']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"❌ Missing columns: {missing}")
                return None
            
            return df
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def extract_real_content(self, text):
        """Remove spam/boilerplate from content"""
        if pd.isna(text) or not str(text).strip():
            return ""
        
        t = str(text).lower()
        
        # Remove URLs
        t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
        t = re.sub(r'\s+', ' ', t)
        
        # Cut at spam markers
        markers = [
            'topcv - tiếp lợi thế',
            'chọn đúng việc',
            'tải app topcv',
            'việc làm mới nhất',
            'cẩm nang nghề nghiệp',
            'chính sách bảo mật',
        ]
        
        positions = [t.find(m) for m in markers if t.find(m) != -1]
        
        # Detect repetitive spam
        spam = re.search(r'(?:\bviệc\s+làm\b.{0,40}){15,}', t)
        if spam:
            positions.append(spam.start())
        
        if positions:
            cut_at = min(positions)
            if cut_at > 0:
                t = t[:cut_at]
        
        # If too short, try to extract from anchors
        if len(t) < 800:
            full = str(text).lower()
            anchors = ['mô tả công việc', 'job description', 'yêu cầu ứng viên', 'quyền lợi']
            starts = [full.find(a) for a in anchors if full.find(a) != -1]
            if starts:
                start = min(starts)
                t2 = full[start:start+25000]
                t2 = re.sub(r'https?://\S+', ' ', t2)
                t2 = re.sub(r'\s+', ' ', t2)
                if len(t2) > len(t):
                    t = t2
        
        return t.strip()
    
    def detect_job_group(self, title):
        """Classify job group"""
        if pd.isna(title):
            return 'other'
        
        title = str(title).lower()
        
        for group, pattern in JOB_GROUP_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                return group
        
        return 'other'
    
    def extract_salary(self, title, content):
        """Extract salary (min, max, avg) in million VND"""
        if pd.isna(title):
            title = ""
        if pd.isna(content):
            content = ""
        
        combined = (str(title) + " " + self.extract_real_content(content))[:6000].lower()
        
        # Check negotiable
        if any(kw in combined for kw in ['thỏa thuận', 'thoả thuận', 'negotiate']):
            return (None, None, None)
        
        # VND range
        vnd_range = re.search(r'(\d+(?:[\.,]\d+)?)\s*[-~đến]+\s*(\d+(?:[\.,]\d+)?)\s*(?:triệu|tr)\b', combined)
        if vnd_range:
            a = float(vnd_range.group(1).replace(',', '.'))
            b = float(vnd_range.group(2).replace(',', '.'))
            return (a, b, (a + b) / 2)
        
        # VND single
        vnd_single = re.search(r'(\d+(?:[\.,]\d+)?)\s*(?:triệu|tr)\b', combined)
        if vnd_single:
            val = float(vnd_single.group(1).replace(',', '.'))
            if 5 <= val <= 200:  # Sanity check
                return (val, val, val)
        
        # USD range (convert to VND: 1 USD ≈ 0.024M VND)
        usd_range = re.search(r'(\d+(?:[\.,]\d+)?)\s*[-~]+\s*(\d+(?:[\.,]\d+)?)\s*usd', combined)
        if usd_range:
            a = float(usd_range.group(1).replace(',', '.')) * 0.024
            b = float(usd_range.group(2).replace(',', '.')) * 0.024
            return (a, b, (a + b) / 2)
        
        return (None, None, None)
    
    def extract_experience(self, title, content):
        """Extract experience in years"""
        if pd.isna(title):
            title = ""
        if pd.isna(content):
            content = ""
        
        combined = (str(title) + " " + self.extract_real_content(content))[:6000].lower()
        
        # Fresher
        fresher = [r'\bfresher\b', r'fresh\s+graduate', r'không\s+yêu\s+cầu\s+kinh\s+nghiệm']
        if any(re.search(p, combined) for p in fresher):
            return 0.0
        
        # Range
        range_match = re.search(r'(\d+)\s*[-~đến]+\s*(\d+)\s*(?:năm|years?)', combined)
        if range_match:
            a = float(range_match.group(1))
            b = float(range_match.group(2))
            if 0 <= a <= 20 and 0 <= b <= 20:
                return (a + b) / 2
        
        # Plus
        plus_match = re.search(r'(\d+)\+\s*(?:năm|years?)', combined)
        if plus_match:
            val = float(plus_match.group(1))
            if 0 <= val <= 20:
                return val
        
        # Single
        single = re.search(r'(\d+)\s*(?:năm|years?)', combined)
        if single:
            val = float(single.group(1))
            if 0 <= val <= 20:
                return val
        
        return None
    
    def extract_skills(self, title, content):
        """Extract skills (context-aware to reduce false positives)"""
        if pd.isna(title):
            title = ""
        if pd.isna(content):
            content = ""
        
        title_lower = str(title).lower()
        content_clean = self.extract_real_content(content)
        
        # Use first 40k chars to avoid spam
        combined = f"{title_lower} {content_clean}"[:40000]
        
        detected = []
        for skill, pattern in self.skill_patterns.items():
            if re.search(pattern, combined, re.IGNORECASE):
                detected.append(skill)
        
        return detected
    
    def extract_features(self, df):
        """Main feature extraction pipeline"""
        print("\n" + "="*60)
        print("🔧 EXTRACTING FEATURES")
        print("="*60)
        
        # 1. Job group
        print("\n   1/5: Job group...")
        df['job_group'] = df['tieu_de_clean'].apply(self.detect_job_group)
        groups = df['job_group'].value_counts()
        print(f"      Groups: {len(groups)}")
        print(f"      'other': {groups.get('other', 0)}/{len(df)} ({groups.get('other', 0)/len(df)*100:.1f}%)")
        
        # 2. Salary
        print("\n   2/5: Salary...")
        salary = df.apply(lambda r: self.extract_salary(r['tieu_de_clean'], r['noi_dung_clean']), axis=1)
        df['salary_min'] = salary.apply(lambda x: x[0])
        df['salary_max'] = salary.apply(lambda x: x[1])
        df['salary_avg'] = salary.apply(lambda x: x[2])
        n_sal = df['salary_avg'].notna().sum()
        print(f"      Detected: {n_sal}/{len(df)} ({n_sal/len(df)*100:.1f}%)")
        if n_sal > 0:
            print(f"      Avg: {df['salary_avg'].mean():.1f}M VND")
        
        # 3. Experience
        print("\n   3/5: Experience...")
        df['experience_years'] = df.apply(lambda r: self.extract_experience(r['tieu_de_clean'], r['noi_dung_clean']), axis=1)
        n_exp = df['experience_years'].notna().sum()
        print(f"      Detected: {n_exp}/{len(df)} ({n_exp/len(df)*100:.1f}%)")
        if n_exp > 0:
            print(f"      Avg: {df['experience_years'].mean():.1f} years")
        
        # 4. Skills
        print("\n   4/5: Skills (context-aware)...")
        df['skills'] = df.apply(lambda r: self.extract_skills(r['tieu_de_clean'], r['noi_dung_clean']), axis=1)
        df['n_skills'] = df['skills'].apply(len)
        df['skills_str'] = df['skills'].apply(lambda lst: ', '.join(lst))
        
        avg_skills = df['n_skills'].mean()
        print(f"      Avg skills/job: {avg_skills:.2f}")
        
        # Top skills
        all_skills = []
        for skills in df['skills']:
            all_skills.extend(skills)
        
        if all_skills:
            skill_counts = Counter(all_skills)
            print(f"\n      Top 10 skills:")
            for skill, count in skill_counts.most_common(10):
                pct = count / len(df) * 100
                print(f"         {skill:15s}: {count:4d} ({pct:5.1f}%)")
        
        # 5. Content metrics
        print("\n   5/5: Content metrics...")
        df['content_real_length'] = df['noi_dung_clean'].apply(lambda x: len(self.extract_real_content(x)))
        print(f"      Avg content: {df['content_real_length'].mean():.0f} chars")
        
        return df
    
    def save_features(self, df, source):
        """Save features"""
        _, features_path = self.get_paths(source)
        
        print("\n" + "="*60)
        print("💾 SAVING")
        print("="*60)
        
        features_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(features_path, index=False, encoding='utf-8-sig')
        
        size_mb = features_path.stat().st_size / (1024 * 1024)
        
        print(f"\n   ✅ Saved: {features_path}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Size: {size_mb:.2f} MB")
    
    def print_summary(self, df):
        """Print summary"""
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        
        print(f"\n   Total jobs: {len(df):,}")
        print(f"\n   Features:")
        print(f"      Job groups: {df['job_group'].nunique()}")
        print(f"      With salary: {df['salary_avg'].notna().sum()} ({df['salary_avg'].notna().sum()/len(df)*100:.1f}%)")
        print(f"      With experience: {df['experience_years'].notna().sum()} ({df['experience_years'].notna().sum()/len(df)*100:.1f}%)")
        print(f"      Avg skills: {df['n_skills'].mean():.2f}")
        
        print(f"\n   Sample (first 3):")
        cols = ['tieu_de', 'job_group', 'n_skills', 'salary_avg', 'experience_years']
        available = [c for c in cols if c in df.columns]
        print(df[available].head(3).to_string(index=False))
    
    def process(self, source):
        """Process single source"""
        print("\n" + "="*60)
        print(f"🔄 PROCESSING {source.upper()}")
        print("="*60)
        
        # Load
        df = self.load_data(source)
        if df is None:
            return False
        
        # Extract
        df = self.extract_features(df)
        
        # Save
        self.save_features(df, source)
        
        # Summary
        self.print_summary(df)
        
        return True
    
    def run(self):
        """Run extraction"""
        print("\n" + "="*60)
        print("🔧 FEATURE EXTRACTION")
        print("="*60)
        print(f"\n Source: {self.source.upper()}")
        
        if self.source == 'both':
            for src in ['topcv', 'vietnamworks']:
                success = self.process(src)
                if not success:
                    print(f"\n⚠️  Skipping {src.upper()}")
        else:
            success = self.process(self.source)
            if not success:
                return False
        
        print("\n" + "="*60)
        print("✅ COMPLETED")
        print("="*60)
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Extract features from cleaned job data',
        epilog='Example: python3 src/04_extract_features.py --source vietnamworks'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        choices=['topcv', 'vietnamworks', 'both'],
        default='topcv',
        help='Data source (default: topcv)'
    )
    
    args = parser.parse_args()
    
    extractor = FeatureExtractor(source=args.source)
    success = extractor.run()
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()