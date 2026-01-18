"""
config.py

Cấu hình toàn bộ project TopCV Analysis.
Based on: Code cũ (simple naming conventions)
"""

from pathlib import Path

# ============================================================
# BASE SETTINGS
# ============================================================

BASE_URL = "https://www.topcv.vn"
START_URL = "https://www.topcv.vn/tim-viec-lam-it"

# ============================================================
# FILE PATHS
# ============================================================

# Raw data (từ scraping)
RAW_LIST_PATH = "data/raw/topcv_it_list.csv"
RAW_DETAIL_PATH = "data/raw/topcv_it_detail.csv"
FAILED_LINKS_PATH = "data/raw/failed_links.txt"

# Processed data
CLEAN_PATH = "data/processed/topcv_it_clean.csv"
FEATURES_PATH = "data/processed/topcv_it_features.csv"
RULES_PATH = "data/processed/topcv_skill_rules.csv"

# Reports
FIG_DIR = "reports/figures"
SUMMARY_STATS_PATH = "reports/figures/summary_stats.txt"
ASSOCIATION_REPORT_PATH = "data/processed/association_rules_report.txt"

# ============================================================
# SCRAPING SETTINGS
# ============================================================

# Limits
MAX_JOBS = 500
MAX_PAGES = 200

# Network
MAX_RETRIES = 3
TIMEOUT = 30

# Rate limiting
RATE_LIMIT_SLEEP = (2.0, 5.0)  # Sleep 2-5 giây random

# Checkpoint
CHECKPOINT_INTERVAL = 10  # Save mỗi 10 items/pages

# Headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

# Multiple headers cho rotation
HEADERS_LIST = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "vi-VN,vi;q=0.9",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "vi-VN,vi;q=0.9",
    },
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "vi-VN,vi;q=0.9",
    },
]

# ============================================================
# DATA CLEANING SETTINGS
# ============================================================

MIN_CONTENT_LENGTH = 200  # Minimum chars để filter errors

# ============================================================
# FEATURE EXTRACTION - SKILLS
# ============================================================

# Từ code cũ - 25 skills phổ biến
SKILL_PATTERNS = {
    # Programming languages
    "python": r"\bpython\b",
    "java": r"\bjava\b",
    "c++": r"\bc\+\+\b",
    "c#": r"\bc\#\b",
    "javascript": r"\bjavascript\b|\bjs\b",
    "typescript": r"\btypescript\b|\bts\b",
    "php": r"\bphp\b",
    
    # Frontend
    "react": r"\breact\b|\breactjs\b",
    "nodejs": r"\bnode\.?js\b|\bnodejs\b",
    
    # Backend
    ".net": r"\b\.net\b|\bdotnet\b",
    
    # Database
    "sql": r"\bsql\b",
    "mysql": r"\bmysql\b",
    "postgresql": r"\bpostgres\b|\bpostgresql\b",
    "mongodb": r"\bmongodb\b|\bmongo\b",
    
    # DevOps & Cloud
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "aws": r"\baws\b|\bamazon web services\b",
    "azure": r"\bazure\b",
    "gcp": r"\bgcp\b|\bgoogle cloud\b",
    
    # Tools
    "git": r"\bgit\b|\bgithub\b|\bgitlab\b",
    "linux": r"\blinux\b|\bubuntu\b|\bcentos\b",
    
    # Web
    "html": r"\bhtml\b",
    "css": r"\bcss\b",
    
    # BI/Analytics
    "excel": r"\bexcel\b",
    "powerbi": r"\bpower\s?bi\b",
    "tableau": r"\btableau\b",
}

# ============================================================
# FEATURE EXTRACTION - JOB GROUPS
# ============================================================

# Từ code cũ - Regex patterns (flexible)
JOB_GROUP_RULES = [
    ("data", r"\bdata\b|\banalyst\b|\bai\b|\bml\b|\bmachine learning\b|\bds\b"),
    ("backend", r"\bbackend\b|\bback-end\b|\bapi\b"),
    ("frontend", r"\bfrontend\b|\bfront-end\b|\breact\b|\bvue\b|\bangular\b"),
    ("fullstack", r"\bfullstack\b|\bfull-stack\b"),
    ("devops", r"\bdevops\b|\bsre\b|\bcloud\b|\baws\b|\bkubernetes\b"),
    ("qa", r"\bqa\b|\btester\b|\btest\b"),
    ("mobile", r"\bandroid\b|\bios\b|\bflutter\b|\breact native\b"),
]

DEFAULT_JOB_GROUP = "other"

# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

FIGURE_SIZE = (12, 6)
FIGURE_SIZE_LARGE = (14, 8)
FIGURE_DPI = 300

COLOR_PALETTE = 'viridis'
FONT_FAMILY = ['DejaVu Sans', 'Arial', 'sans-serif']
FONT_SIZE = 10

# ============================================================
# ASSOCIATION RULES SETTINGS
# ============================================================

MIN_SUPPORT = 0.03      # 3% - xuất hiện trong ít nhất 15/500 jobs
MIN_CONFIDENCE = 0.3    # 30% - nếu có A thì có 30% khả năng có B
MIN_LIFT = 1.0          # Chỉ giữ positive correlations
MAX_RULES = 100         # Giữ top 100 rules

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_dirs():
    """Tạo directories nếu chưa tồn tại"""
    dirs = [
        "data/raw",
        "data/processed",
        "reports/figures",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def print_config():
    """Print configuration summary"""
    print("=" * 60)
    print("PROJECT CONFIGURATION")
    print("=" * 60)
    
    print("\n📁 PATHS:")
    print(f"  RAW_LIST_PATH:    {RAW_LIST_PATH}")
    print(f"  RAW_DETAIL_PATH:  {RAW_DETAIL_PATH}")
    print(f"  CLEAN_PATH:       {CLEAN_PATH}")
    print(f"  FEATURES_PATH:    {FEATURES_PATH}")
    print(f"  FIG_DIR:          {FIG_DIR}")
    
    print("\n⚙️ SCRAPING:")
    print(f"  MAX_JOBS:         {MAX_JOBS}")
    print(f"  MAX_PAGES:        {MAX_PAGES}")
    print(f"  RATE_LIMIT:       {RATE_LIMIT_SLEEP}s")
    
    print("\n🔧 FEATURES:")
    print(f"  Skills:           {len(SKILL_PATTERNS)}")
    print(f"  Job groups:       {len(JOB_GROUP_RULES)}")
    
    print("\n🔗 ASSOCIATION RULES:")
    print(f"  MIN_SUPPORT:      {MIN_SUPPORT} ({MIN_SUPPORT*100}%)")
    print(f"  MIN_CONFIDENCE:   {MIN_CONFIDENCE} ({MIN_CONFIDENCE*100}%)")
    print(f"  MAX_RULES:        {MAX_RULES}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test
    print_config()
    
    # Ensure directories exist
    ensure_dirs()
    print("\n✅ Directories created!")