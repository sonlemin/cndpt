from pathlib import Path

BASE_URL = "https://www.topcv.vn"
START_URL = "https://www.topcv.vn/tim-viec-lam-it"

RAW_LIST_PATH = "data/raw/topcv_it_list.csv"
RAW_DETAIL_PATH = "data/raw/topcv_it_detail.csv"
FAILED_LINKS_PATH = "data/raw/failed_links.txt"

CLEAN_PATH = "data/processed/topcv_it_clean.csv"
FEATURES_PATH = "data/processed/topcv_it_features.csv"
RULES_PATH = "data/processed/topcv_skill_rules.csv"

FIG_DIR = "reports/figures"
SUMMARY_STATS_PATH = "reports/figures/summary_stats.txt"
ASSOCIATION_REPORT_PATH = "data/processed/association_rules_report.txt"

MAX_JOBS = 500
MAX_PAGES = 200
MAX_RETRIES = 3
TIMEOUT = 30
RATE_LIMIT_SLEEP = (2.0, 5.0)
CHECKPOINT_INTERVAL = 10
MIN_CONTENT_LENGTH = 200

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
    "react": r"\breact\b|\breactjs\b",
    "vue": r"\bvue\b|\bvuejs\b",
    "angular": r"\bangular\b",
    "html": r"\bhtml\b",
    "css": r"\bcss\b",
    "sass": r"\bsass\b|\bscss\b",
    "nodejs": r"\bnode\.?js\b|\bnodejs\b",
    ".net": r"\b\.net\b|\bdotnet\b",
    "spring": r"\bspring\b(?:\s+boot)?",
    "django": r"\bdjango\b",
    "laravel": r"\blaravel\b",
    "sql": r"\bsql\b",
    "mysql": r"\bmysql\b",
    "postgresql": r"\bpostgres\b|\bpostgresql\b",
    "mongodb": r"\bmongodb\b|\bmongo\b",
    "redis": r"\bredis\b",
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "aws": r"\baws\b|\bamazon web services\b",
    "azure": r"\bazure\b",
    "gcp": r"\bgcp\b|\bgoogle cloud\b",
    "jenkins": r"\bjenkins\b",
    "git": r"\bgit\b|\bgithub\b|\bgitlab\b",
    "linux": r"\blinux\b|\bubuntu\b|\bcentos\b",
    "jira": r"\bjira\b",
    "excel": r"\bexcel\b",
    "powerbi": r"\bpower\s?bi\b",
    "tableau": r"\btableau\b",
    "android": r"\bandroid\b",
    "ios": r"\bios\b",
}

JOB_GROUP_RULES = [
    ("fullstack", r"\b(?:fullstack|full-stack|full stack)\b"),
    ("mobile", r"\b(?:android|ios|mobile|flutter|react native|swift|objective-c|xamarin|ionic)\b"),
    ("game", r"\b(?:game|unity|unreal|godot|gaming)\b"),
    ("intern", r"\b(?:intern|internship|thực tập|thuc tap|fresher|junior|trainee)\b"),
    ("designer", r"\b(?:designer|thiết kế|ui/ux|ux/ui|graphic|figma|photoshop|design)\b"),
    ("business", r"\b(?:business analyst|ba|product owner|product manager|project manager|pm|scrum master)\b"),
    ("data", r"\b(?:data|analyst|analysis|ai|machine learning|ml|deep learning|scientist|etl|big data|spark|bi|powerbi|tableau)\b"),
    ("backend",
     r"\b(?:backend|server|api|java|php|python|django|flask|spring|\.net|nodejs|express|golang|ruby|rails)\b"),
    ("frontend",
     r"\b(?:frontend|react|vue|angular|html|css|sass|bootstrap|tailwind|web developer)\b"),
    ("devops",
     r"\b(?:devops|sre|cloud|aws|azure|gcp|docker|kubernetes|terraform|ansible|sysadmin)\b"),
    ("qa",
     r"\b(?:qa|qc|test|tester|testing|selenium|automation)\b"),
    ("security",
     r"\b(?:security|cyber|pentest|infosec|soc|vulnerability)\b"),
    ("software_engineer",
     r"\b(?:software|developer|engineer|lập trình|programmer|coder|it)\b"),
]

DEFAULT_JOB_GROUP = "other"

MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.3
MIN_LIFT = 1.0
MAX_RULES = 100

def ensure_dirs():
    for d in ["data/raw", "data/processed", "reports/figures"]:
        Path(d).mkdir(parents=True, exist_ok=True)
