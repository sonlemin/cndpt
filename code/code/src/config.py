"""
config_v2.py

Config VERSION 2 - PATTERNS MẠNH HƠN để giảm "other" xuống dưới 20%

CHANGES FROM v1:
- Patterns tổng quát hơn, flexible hơn
- Better ordering (most specific → most general)
- Catch more edge cases
- Better Vietnamese support

IMPROVEMENTS:
- JOB_GROUP_RULES: Thêm nhiều variations, synonyms
- Thứ tự: fullstack → mobile → specific → generic
- Default "software_engineer" rộng hơn

EXPECTED RESULT:
- "other" < 20% (hiện tại: 33%)
- Better coverage for Vietnamese job titles
"""

from pathlib import Path

# ============================================================
# BASE SETTINGS (giống cũ)
# ============================================================

BASE_URL = "https://www.topcv.vn"
START_URL = "https://www.topcv.vn/tim-viec-lam-it"

# ============================================================
# FILE PATHS (giống cũ)
# ============================================================

RAW_LIST_PATH = "data/raw/topcv_it_list.csv"
RAW_DETAIL_PATH = "data/raw/topcv_it_detail.csv"
FAILED_LINKS_PATH = "data/raw/failed_links.txt"

CLEAN_PATH = "data/processed/topcv_it_clean.csv"
FEATURES_PATH = "data/processed/topcv_it_features.csv"
RULES_PATH = "data/processed/topcv_skill_rules.csv"

FIG_DIR = "reports/figures"
SUMMARY_STATS_PATH = "reports/figures/summary_stats.txt"
ASSOCIATION_REPORT_PATH = "data/processed/association_rules_report.txt"

# ============================================================
# SCRAPING SETTINGS (giống cũ)
# ============================================================

MAX_JOBS = 500
MAX_PAGES = 200
MAX_RETRIES = 3
TIMEOUT = 30
RATE_LIMIT_SLEEP = (2.0, 5.0)
CHECKPOINT_INTERVAL = 10

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

MIN_CONTENT_LENGTH = 200

# ============================================================
# FEATURE EXTRACTION - SKILLS (giữ nguyên 40 skills)
# ============================================================

SKILL_PATTERNS = {
    # Programming languages
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
    
    # Frontend
    "react": r"\breact\b|\breactjs\b",
    "vue": r"\bvue\b|\bvuejs\b",
    "angular": r"\bangular\b",
    "html": r"\bhtml\b",
    "css": r"\bcss\b",
    "sass": r"\bsass\b|\bscss\b",
    
    # Backend/Framework
    "nodejs": r"\bnode\.?js\b|\bnodejs\b",
    ".net": r"\b\.net\b|\bdotnet\b",
    "spring": r"\bspring\b(?:\s+boot)?",
    "django": r"\bdjango\b",
    "laravel": r"\blaravel\b",
    
    # Database
    "sql": r"\bsql\b",
    "mysql": r"\bmysql\b",
    "postgresql": r"\bpostgres\b|\bpostgresql\b",
    "mongodb": r"\bmongodb\b|\bmongo\b",
    "redis": r"\bredis\b",
    
    # DevOps & Cloud
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "aws": r"\baws\b|\bamazon web services\b",
    "azure": r"\bazure\b",
    "gcp": r"\bgcp\b|\bgoogle cloud\b",
    "jenkins": r"\bjenkins\b",
    
    # Tools
    "git": r"\bgit\b|\bgithub\b|\bgitlab\b",
    "linux": r"\blinux\b|\bubuntu\b|\bcentos\b",
    "jira": r"\bjira\b",
    
    # BI/Analytics
    "excel": r"\bexcel\b",
    "powerbi": r"\bpower\s?bi\b",
    "tableau": r"\btableau\b",
    
    # Mobile
    "android": r"\bandroid\b",
    "ios": r"\bios\b",
}

# ============================================================
# FEATURE EXTRACTION - JOB GROUPS (VERSION 2 - MẠNH HƠN)
# ============================================================

# STRATEGY v2:
# 1. Most specific first (fullstack, mobile)
# 2. Then role-based (intern, designer, business)
# 3. Then tech-specific (backend với languages, frontend với frameworks)
# 4. Then generic developer/engineer (CATCH-ALL at the end)
# 5. DevOps, QA, Security
# 6. Default "other"

JOB_GROUP_RULES = [
    # ===== LEVEL 1: MOST SPECIFIC (check these first) =====
    
    # 1. Fullstack (must check before backend/frontend)
    ("fullstack", r"\b(?:fullstack|full-stack|full stack)\b"),
    
    # 2. Mobile (very specific)
    ("mobile", r"\b(?:android|ios|mobile|flutter|react native|swift|objective-c|xamarin|ionic)\b"),
    
    # 3. Game (very specific)
    ("game", r"\b(?:game|unity|unreal|godot|gaming)\b"),
    
    # ===== LEVEL 2: ROLE-BASED (before tech-specific) =====
    
    # 4. Intern/Fresher (catch early careers)
    ("intern", r"\b(?:intern|internship|thực tập|thuc tap|fresher|mới tốt nghiệp|graduate|junior|jr|trainee)\b"),
    
    # 5. Designer (UI/UX)
    ("designer", r"\b(?:designer|thiết kế|thiet ke|ui/ux|ux/ui|ux|ui designer|graphic|illustrator|photoshop|figma|sketch|adobe|design)\b"),
    
    # 6. Business Analyst / PM / PO
    ("business", r"\b(?:business analyst|ba|product owner|po|product manager|pm|project manager|quản lý dự án|scrum master|agile coach|program manager)\b"),
    
    # ===== LEVEL 3: TECH-SPECIFIC =====
    
    # 7. Data & AI (before generic analyst)
    ("data", r"\b(?:data|dữ liệu|du lieu|analyst|analysis|phân tích|phan tich|ai|artificial intelligence|machine learning|ml|deep learning|dl|scientist|ds|data engineer|de|etl|data mining|big data|hadoop|spark|bi|business intelligence|tableau|powerbi)\b"),
    
    # 8. Backend (with language specifics)
    ("backend", 
     r"\b(?:"
     r"backend|back-end|back end|server side|api|rest|restful|microservice|"  # generic backend
     r"java developer|java engineer|java programmer|"  # Java
     r"php developer|php engineer|php programmer|laravel|"  # PHP
     r"python developer|python engineer|django|flask|"  # Python
     r"\.net developer|\.net engineer|c# developer|"  # .NET/C#
     r"nodejs developer|node developer|express|"  # Node.js
     r"golang|go developer|"  # Go
     r"ruby developer|rails|"  # Ruby
     r"spring boot|spring framework"  # Frameworks
     r")\b"),
    
    # 9. Frontend (with framework specifics)
    ("frontend", 
     r"\b(?:"
     r"frontend|front-end|front end|client side|"  # generic frontend
     r"reactjs|react developer|react engineer|"  # React
     r"vuejs|vue developer|"  # Vue
     r"angular developer|angular|angularjs|"  # Angular
     r"web developer|web designer|web programmer|"  # Web
     r"ui developer|interface developer|"  # UI
     r"html|css|sass|scss|less|bootstrap|tailwind"  # Technologies
     r")\b"),
    
    # 10. DevOps / Cloud / Infrastructure
    ("devops", 
     r"\b(?:"
     r"devops|devsecops|sre|site reliability|"  # DevOps/SRE
     r"cloud engineer|cloud architect|"  # Cloud
     r"infrastructure|infra|platform engineer|"  # Infrastructure
     r"aws|azure|gcp|google cloud|amazon web services|"  # Cloud providers
     r"kubernetes|k8s|docker|container|"  # Container tech
     r"jenkins|gitlab ci|github actions|travis|circleci|"  # CI/CD
     r"terraform|ansible|puppet|chef|"  # IaC
     r"system administrator|sysadmin|sys admin|quản trị hệ thống"  # SysAdmin
     r")\b"),
    
    # 11. QA / Test
    ("qa", 
     r"\b(?:"
     r"qa|qc|quality assurance|quality control|"  # QA/QC
     r"test|tester|testing|kiểm thử|kiem thu|"  # Test
     r"automation test|manual test|"  # Test types
     r"selenium|appium|cypress|jest|mocha|"  # Test tools
     r"performance test|load test|security test"  # Test categories
     r")\b"),
    
    # 12. Security
    ("security", 
     r"\b(?:"
     r"security|bảo mật|bao mat|an ninh mạng|"  # Security
     r"penetration|pentest|ethical hacker|white hat|"  # PenTest
     r"cyber|cybersecurity|infosec|appsec|"  # Cyber
     r"vulnerability|exploit|"  # Vuln
     r"soc|security operations center"  # SOC
     r")\b"),
    
    # ===== LEVEL 4: GENERIC CATCH-ALL =====
    
    # 13. Generic Software Engineer/Developer/Programmer
    # This should catch anything with "developer", "engineer", "programmer"
    # that didn't match more specific patterns above
    ("software_engineer", 
     r"\b(?:"
     r"software engineer|software developer|phần mềm|phan mem|"  # Software
     r"developer|dev|lập trình|lap trinh|lập trình viên|"  # Developer
     r"engineer|kỹ sư|ky su|"  # Engineer
     r"programmer|coder|coding|"  # Programmer
     r"it engineer|it developer|"  # IT
     r"technical|tech lead|team lead"  # Lead
     r")\b"),
    
    # ===== FALLBACK: "other" =====
    # Any job that doesn't match above patterns will be "other"
]

DEFAULT_JOB_GROUP = "other"

# ============================================================
# VISUALIZATION SETTINGS (giống cũ)
# ============================================================

FIGURE_SIZE = (12, 6)
FIGURE_SIZE_LARGE = (14, 8)
FIGURE_DPI = 300
COLOR_PALETTE = 'viridis'
FONT_FAMILY = ['DejaVu Sans', 'Arial', 'sans-serif']
FONT_SIZE = 10

# ============================================================
# ASSOCIATION RULES SETTINGS (giống cũ)
# ============================================================

MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.3
MIN_LIFT = 1.0
MAX_RULES = 100

# ============================================================
# UTILITY FUNCTIONS (giống cũ)
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
    print("PROJECT CONFIGURATION (VERSION 2)")
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
    
    print("\n📗 ASSOCIATION RULES:")
    print(f"  MIN_SUPPORT:      {MIN_SUPPORT} ({MIN_SUPPORT*100}%)")
    print(f"  MIN_CONFIDENCE:   {MIN_CONFIDENCE} ({MIN_CONFIDENCE*100}%)")
    print(f"  MAX_RULES:        {MAX_RULES}")
    
    print("\n✨ VERSION 2 IMPROVEMENTS:")
    print("  ✅ Patterns tổng quát hơn, flexible hơn")
    print("  ✅ Better ordering: specific → generic")
    print("  ✅ Catch more Vietnamese variations")
    print("  ✅ Expanded backend patterns (Java, PHP, Python, .NET, Go, Ruby)")
    print("  ✅ Expanded frontend patterns (React, Vue, Angular)")
    print("  ✅ Generic 'software_engineer' as catch-all")
    print("  🎯 Target: 'other' < 20% (currently: ~33%)")
    
    print("=" * 60)


if __name__ == "__main__":
    print_config()
    ensure_dirs()
    print("\n✅ Directories created!")