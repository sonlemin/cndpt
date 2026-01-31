# src/config_vietnamworks.py
"""
Configuration for VietnamWorks Scraper

Contains all settings specific to VietnamWorks scraping,
separate from TopCV config to keep things organized.
"""

from pathlib import Path

# ============================================================================
# VietnamWorks URLs
# ============================================================================
BASE_URL_VNW = "https://www.vietnamworks.com"

# Search URL for IT jobs
START_URL = "https://www.vietnamworks.com/viec-lam?q=it&sorting=relevant"

# Alternative URLs you can use:
# START_URL = "https://www.vietnamworks.com/it-phan-mem-kv"  # IT Software category
# START_URL = "https://www.vietnamworks.com/viec-lam?g=5"  # IT category (g=5)

# ============================================================================
# File Paths - VietnamWorks Data
# ============================================================================
RAW_LIST_PATH_VNW = "data/raw/vietnamworks_it_list.csv"
RAW_DETAIL_PATH_VNW = "data/raw/vietnamworks_it_detail.csv"

CLEAN_PATH_VNW = "data/processed/vietnamworks_it_clean.csv"
FEATURES_PATH_VNW = "data/processed/vietnamworks_it_features.csv"

# ============================================================================
# Scraping Settings
# ============================================================================
MAX_JOBS_VNW = 2000  # Target number of jobs to scrape (increased from 500)
MAX_SCROLLS = 20     # Maximum scrolls on search page (infinite scroll)
SCROLL_PAUSE_TIME = 2  # Seconds to wait between scrolls

# Selenium settings
HEADLESS = True  # Run browser in headless mode (no UI)
TIMEOUT = 30     # Page load timeout (seconds)

# Checkpoint & retry
CHECKPOINT_INTERVAL = 10  # Save every N jobs
MAX_RETRIES = 2

# ============================================================================
# Reuse from TopCV Config
# ============================================================================
# These are the same for both sources, so we import from main config
try:
    from config import (
        SKILL_PATTERNS,
        JOB_GROUP_RULES,
        FIG_DIR,
        MIN_SUPPORT,
        MIN_CONFIDENCE,
        MIN_LIFT,
        MAX_RULES,
    )
except ImportError:
    # Fallback if config.py not available
    print("⚠️  Could not import from config.py, using defaults")
    
    SKILL_PATTERNS = {
        "python": r"\bpython\b",
        "java": r"\bjava\b(?!\s*script)",
        "javascript": r"\bjavascript\b|\bjs\b",
        # ... (add more as needed)
    }
    
    JOB_GROUP_RULES = [
        ("fullstack", r"\b(?:fullstack|full-stack|full stack)\b"),
        ("mobile", r"\b(?:android|ios|mobile)\b"),
        # ... (add more as needed)
    ]
    
    FIG_DIR = "reports/figures"
    MIN_SUPPORT = 0.03
    MIN_CONFIDENCE = 0.3
    MIN_LIFT = 1.0
    MAX_RULES = 100

# ============================================================================
# Helper Functions
# ============================================================================
def ensure_dirs_vnw():
    """Create necessary directories for VietnamWorks data"""
    for d in ["data/raw", "data/processed", "reports/figures"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ VietnamWorks directories created")


if __name__ == "__main__":
    # Test config
    print("VietnamWorks Scraper Configuration")
    print("=" * 50)
    print(f"Base URL: {BASE_URL_VNW}")
    print(f"Start URL: {START_URL}")
    print(f"Max jobs: {MAX_JOBS_VNW}")
    print(f"Max scrolls: {MAX_SCROLLS}")
    print(f"Raw list path: {RAW_LIST_PATH_VNW}")
    print(f"Raw detail path: {RAW_DETAIL_PATH_VNW}")
    print("=" * 50)
    
    ensure_dirs_vnw()