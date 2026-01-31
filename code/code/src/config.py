"""
Unified Configuration for IT Job Scraper & Analysis

Supports both TopCV and VietnamWorks sources.
Used by all scripts in the pipeline.
"""

from pathlib import Path

# ============================================================================
# DIRECTORIES
# ============================================================================
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = Path("fig")
REPORTS_DIR = Path("reports")

# Create directories
for dir_path in [RAW_DIR, PROCESSED_DIR, FIG_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TOPCV CONFIGURATION
# ============================================================================
# URLs
BASE_URL = "https://www.topcv.vn"
START_URL = "https://www.topcv.vn/tim-viec-lam-it"

# File paths
TOPCV_LIST_PATH = RAW_DIR / "topcv_it_list.csv"
TOPCV_DETAIL_PATH = RAW_DIR / "topcv_it_detail.csv"
TOPCV_CLEAN_PATH = PROCESSED_DIR / "topcv_it_clean.csv"
TOPCV_FEATURES_PATH = PROCESSED_DIR / "topcv_it_features.csv"
TOPCV_FAILED_LINKS = RAW_DIR / "topcv_failed_links.txt"

# Scraping settings
MAX_JOBS = 500
MAX_PAGES = 200
MAX_RETRIES = 3
TIMEOUT = 30
RATE_LIMIT_SLEEP = (2.0, 5.0)
CHECKPOINT_INTERVAL = 10

# ============================================================================
# VIETNAMWORKS CONFIGURATION
# ============================================================================
# URLs
BASE_URL_VNW = "https://www.vietnamworks.com"
START_URL_VNW = "https://www.vietnamworks.com/viec-lam?q=it&sorting=relevant"

# File paths
VNW_LIST_PATH = RAW_DIR / "vietnamworks_it_list.csv"
VNW_DETAIL_PATH = RAW_DIR / "vietnamworks_it_detail.csv"
VNW_CLEAN_PATH = PROCESSED_DIR / "vietnamworks_it_clean.csv"
VNW_FEATURES_PATH = PROCESSED_DIR / "vietnamworks_it_features.csv"
VNW_FAILED_LINKS = RAW_DIR / "vietnamworks_failed_links.txt"

# Scraping settings (VietnamWorks uses Selenium)
MAX_JOBS_VNW = 2000
MAX_SCROLLS = 20
SCROLL_PAUSE_TIME = 2
HEADLESS = True

# ============================================================================
# MERGED DATA PATHS
# ============================================================================
MERGED_CLEAN_PATH = PROCESSED_DIR / "merged_it_clean.csv"
MERGED_FEATURES_PATH = PROCESSED_DIR / "merged_it_features.csv"

# ============================================================================
# LEGACY COMPATIBILITY (for old scripts)
# ============================================================================
RAW_LIST_PATH = TOPCV_LIST_PATH
RAW_DETAIL_PATH = TOPCV_DETAIL_PATH
CLEAN_PATH = TOPCV_CLEAN_PATH
FEATURES_PATH = TOPCV_FEATURES_PATH
FAILED_LINKS_PATH = TOPCV_FAILED_LINKS

# Report paths (legacy)
SUMMARY_STATS_PATH = REPORTS_DIR / "summary_stats.txt"
ASSOCIATION_REPORT_PATH = PROCESSED_DIR / "association_rules_report.txt"
RULES_PATH = PROCESSED_DIR / "topcv_skill_rules.csv"

# ============================================================================
# SOURCE CONFIGURATIONS (for new multi-source scripts)
# ============================================================================
SOURCES = {
    'topcv': {
        'name': 'TopCV',
        'base_url': BASE_URL,
        'start_url': START_URL,
        'list_path': TOPCV_LIST_PATH,
        'detail_path': TOPCV_DETAIL_PATH,
        'clean_path': TOPCV_CLEAN_PATH,
        'features_path': TOPCV_FEATURES_PATH,
        'failed_links_path': TOPCV_FAILED_LINKS,
        'min_content_length': 800,
        'min_clean_length': 200,
    },
    'vietnamworks': {
        'name': 'VietnamWorks',
        'base_url': BASE_URL_VNW,
        'start_url': START_URL_VNW,
        'list_path': VNW_LIST_PATH,
        'detail_path': VNW_DETAIL_PATH,
        'clean_path': VNW_CLEAN_PATH,
        'features_path': VNW_FEATURES_PATH,
        'failed_links_path': VNW_FAILED_LINKS,
        'min_content_length': 800,
        'min_clean_length': 200,
    }
}

# ============================================================================
# DATA CLEANING PARAMETERS
# ============================================================================
MIN_TITLE_LENGTH = 10           # Minimum title length (chars)
MAX_TITLE_LENGTH = 200          # Maximum title length (chars)
MIN_CONTENT_RAW = 800           # Minimum raw content length
MIN_CONTENT_CLEAN = 200         # Minimum cleaned content length
MIN_CONTENT_LENGTH = 200        # Legacy compatibility
MAX_DUPLICATES_RATIO = 0.3      # Max allowed duplicate ratio

# ============================================================================
# FEATURE EXTRACTION - TOP SKILLS
# ============================================================================
# Top skills for Vietnamese IT market (used by feature extraction)
TOP_SKILLS = [
    # Programming Languages
    'JavaScript', 'Python', 'Java', 'C#', 'PHP', 'TypeScript', 
    'C++', 'Go', 'Ruby', 'Swift', 'Kotlin',
    
    # Frontend Frameworks
    'React', 'Angular', 'Vue', 'Next.js', 'HTML', 'CSS',
    
    # Backend Frameworks
    'Node.js', 'Django', 'Spring', 'Laravel', 'ASP.NET', 'Flask',
    
    # Databases
    'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 'SQL Server',
    
    # Cloud & DevOps
    'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'CI/CD',
    
    # Data & ML
    'SQL', 'Pandas', 'NumPy', 'TensorFlow', 'PyTorch', 'Scikit-learn',
    
    # Mobile
    'iOS', 'Android', 'React Native', 'Flutter',
    
    # Tools & Others
    'Git', 'Jira', 'Agile', 'Scrum',
]

# ============================================================================
# SKILL PATTERNS (for regex matching)
# ============================================================================
SKILL_PATTERNS = {
    "python": r"\bpython\b",
    "java": r"\bjava\b(?!\s*script)",
    "javascript": r"\bjavascript\b|\bjs\b(?!\son)",
    "typescript": r"\btypescript\b|\bts\b",
    "php": r"\bphp\b",
    "c++": r"\bc\+\+\b",
    "c#": r"\bc\#\b|\bc sharp\b",
    "go": r"\b(?:golang|go)\b",
    "ruby": r"\bruby\b",
    "kotlin": r"\bkotlin\b",
    "swift": r"\bswift\b",
    
    "react": r"\breact\b|\breactjs\b",
    "vue": r"\bvue\b|\bvuejs\b",
    "angular": r"\bangular\b",
    "next.js": r"\bnext\.?js\b",
    "html": r"\bhtml\b",
    "css": r"\bcss\b",
    "sass": r"\bsass\b|\bscss\b",
    
    "nodejs": r"\bnode\.?js\b|\bnodejs\b",
    ".net": r"\b\.net\b|\bdotnet\b",
    "spring": r"\bspring\b(?:\s+boot)?",
    "django": r"\bdjango\b",
    "laravel": r"\blaravel\b",
    "flask": r"\bflask\b",
    "asp.net": r"\basp\.net\b",
    
    "sql": r"\bsql\b",
    "mysql": r"\bmysql\b",
    "postgresql": r"\bpostgres\b|\bpostgresql\b",
    "mongodb": r"\bmongodb\b|\bmongo\b",
    "redis": r"\bredis\b",
    "oracle": r"\boracle\b",
    "sql server": r"\bsql\s+server\b",
    
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "aws": r"\baws\b|\bamazon web services\b",
    "azure": r"\bazure\b",
    "gcp": r"\bgcp\b|\bgoogle cloud\b",
    "ci/cd": r"\bci/cd\b|\bcicd\b",
    
    "git": r"\bgit\b(?!hub|lab)",
    "github": r"\bgithub\b",
    "gitlab": r"\bgitlab\b",
    "jira": r"\bjira\b",
    "agile": r"\bagile\b",
    "scrum": r"\bscrum\b",
    
    "linux": r"\blinux\b|\bubuntu\b|\bcentos\b",
    "jenkins": r"\bjenkins\b",
    
    # Mobile
    "android": r"\bandroid\b",
    "ios": r"\bios\b",
    "react native": r"\breact\s+native\b",
    "flutter": r"\bflutter\b",
    
    # Data & BI
    "pandas": r"\bpandas\b",
    "numpy": r"\bnumpy\b",
    "tensorflow": r"\btensorflow\b",
    "pytorch": r"\bpytorch\b",
    "scikit-learn": r"\bscikit-learn\b|\bsklearn\b",
    "excel": r"\bexcel\b",
    "powerbi": r"\bpower\s?bi\b",
    "tableau": r"\btableau\b",
}

# ============================================================================
# JOB GROUP CLASSIFICATION RULES
# ============================================================================
JOB_GROUP_RULES = [
    ("fullstack", r"\b(?:fullstack|full-stack|full stack)\b"),
    ("mobile", r"\b(?:android|ios|mobile|flutter|react native|swift|objective-c|xamarin|ionic)\b"),
    ("game", r"\b(?:game|unity|unreal|godot|gaming)\b"),
    ("intern", r"\b(?:intern|internship|thực tập|thuc tap|fresher|junior|trainee)\b"),
    ("designer", r"\b(?:designer|thiết kế|ui/ux|ux/ui|graphic|figma|photoshop|design)\b"),
    ("business", r"\b(?:business analyst|ba|product owner|product manager|project manager|pm|scrum master)\b"),
    ("data", r"\b(?:data|analyst|analysis|ai|machine learning|ml|deep learning|scientist|etl|big data|spark|bi|powerbi|tableau)\b"),
    ("backend", r"\b(?:backend|server|api|java|php|python|django|flask|spring|\.net|nodejs|express|golang|ruby|rails)\b"),
    ("frontend", r"\b(?:frontend|react|vue|angular|html|css|sass|bootstrap|tailwind|web developer)\b"),
    ("devops", r"\b(?:devops|sre|cloud|aws|azure|gcp|docker|kubernetes|terraform|ansible|sysadmin)\b"),
    ("qa", r"\b(?:qa|qc|test|tester|testing|selenium|automation)\b"),
    ("security", r"\b(?:security|cyber|pentest|infosec|soc|vulnerability)\b"),
    ("software_engineer", r"\b(?:software|developer|engineer|lập trình|programmer|coder|it)\b"),
]

DEFAULT_JOB_GROUP = "other"

# ============================================================================
# REGION MAPPING (Vietnamese cities to regions)
# ============================================================================
REGION_MAPPING = {
    # North (Miền Bắc)
    'hanoi': 'North',
    'hà nội': 'North',
    'ha noi': 'North',
    'hn': 'North',
    'haiphong': 'North',
    'hải phòng': 'North',
    'hai phong': 'North',
    'hp': 'North',
    'thai nguyen': 'North',
    'thái nguyên': 'North',
    'nam dinh': 'North',
    'nam định': 'North',
    
    # Central (Miền Trung)
    'danang': 'Central',
    'đà nẵng': 'Central',
    'da nang': 'Central',
    'dn': 'Central',
    'hue': 'Central',
    'huế': 'Central',
    'quang nam': 'Central',
    'quảng nam': 'Central',
    'quang tri': 'Central',
    'quảng trị': 'Central',
    'binh dinh': 'Central',
    'bình định': 'Central',
    
    # South (Miền Nam)
    'hochiminh': 'South',
    'hồ chí minh': 'South',
    'ho chi minh': 'South',
    'hcm': 'South',
    'saigon': 'South',
    'sài gòn': 'South',
    'binh duong': 'South',
    'bình dương': 'South',
    'dong nai': 'South',
    'đồng nai': 'South',
    'bien hoa': 'South',
    'biên hòa': 'South',
    'vung tau': 'South',
    'vũng tàu': 'South',
    'can tho': 'South',
    'cần thơ': 'South',
    'long an': 'South',
    'ba ria': 'South',
    'bà rịa': 'South',
}

# ============================================================================
# HTTP HEADERS
# ============================================================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

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

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
# Association Rules Mining
MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.3
MIN_LIFT = 1.0
MAX_RULES = 100
ASSOCIATION_RULES_MIN_SUPPORT = 0.05
ASSOCIATION_RULES_MIN_CONFIDENCE = 0.3
ASSOCIATION_RULES_MIN_LIFT = 1.2

# Clustering
CLUSTERING_N_CLUSTERS = 5
CLUSTERING_RANDOM_STATE = 42

# Forecasting
FORECAST_PERIODS = 12  # Months to forecast

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def ensure_dirs():
    """Create all necessary directories"""
    for d in [RAW_DIR, PROCESSED_DIR, FIG_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_source_config(source: str) -> dict:
    """
    Get configuration for a specific source
    
    Args:
        source: 'topcv', 'vietnamworks', or 'merged'
    
    Returns:
        dict: Configuration for the source
    
    Raises:
        ValueError: If source is unknown
    """
    if source == 'merged':
        return {
            'name': 'Merged',
            'clean_path': MERGED_CLEAN_PATH,
            'features_path': MERGED_FEATURES_PATH,
        }
    
    if source not in SOURCES:
        raise ValueError(f"Unknown source: {source}. Available: {list(SOURCES.keys())}")
    
    return SOURCES[source]


def print_config(source: str = None):
    """Print configuration for debugging"""
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    
    if source:
        print(f"\nSource: {source.upper()}")
        config = get_source_config(source)
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print(f"\nAvailable sources: {list(SOURCES.keys())}")
        print(f"Data directory: {DATA_DIR}")
        print(f"Raw directory: {RAW_DIR}")
        print(f"Processed directory: {PROCESSED_DIR}")
        print(f"Figure directory: {FIG_DIR}")
        print(f"Reports directory: {REPORTS_DIR}")
    
    print("="*60)


# ============================================================================
# INITIALIZATION
# ============================================================================
# Create directories on import
ensure_dirs()


if __name__ == "__main__":
    # Test configuration
    print("IT Job Scraper & Analysis - Configuration Test")
    print("="*60)
    
    print("\n📁 Directories:")
    print(f"  Data:      {DATA_DIR}")
    print(f"  Raw:       {RAW_DIR}")
    print(f"  Processed: {PROCESSED_DIR}")
    print(f"  Figures:   {FIG_DIR}")
    print(f"  Reports:   {REPORTS_DIR}")
    
    print("\n📊 Sources:")
    for source_name, source_config in SOURCES.items():
        print(f"\n  {source_name.upper()}:")
        print(f"    Name:         {source_config['name']}")
        print(f"    Base URL:     {source_config['base_url']}")
        print(f"    Clean path:   {source_config['clean_path']}")
        print(f"    Features:     {source_config['features_path']}")
    
    print(f"\n🔧 Skills: {len(TOP_SKILLS)} skills defined")
    print(f"🏷️  Job groups: {len(JOB_GROUP_RULES)} groups")
    print(f"🗺️  Regions: {len(set(REGION_MAPPING.values()))} regions")
    
    print("\n✅ Configuration loaded successfully")