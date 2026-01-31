# src/01_scrape_list_vietnamworks_v2.py
"""
VietnamWorks List Scraper V2 - Pagination Support

VietnamWorks uses both:
1. Infinite scroll (limited)
2. Pagination (page=1, page=2, etc.)

This version scrapes multiple pages to get more jobs.
"""

import pandas as pd
import time
import random
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

from config_vietnamworks import (
    BASE_URL_VNW,
    RAW_LIST_PATH_VNW,
    MAX_JOBS_VNW,
    MAX_SCROLLS,
    SCROLL_PAUSE_TIME,
    HEADLESS,
    TIMEOUT,
)
from utils import normalize_spaces, clean_url, sleep_random


def setup_driver(headless=True):
    """Setup Chrome WebDriver"""
    options = webdriver.ChromeOptions()
    
    if headless:
        options.add_argument('--headless')
    
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36')
    
    # Memory management
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    
    # Disable images to speed up
    prefs = {'profile.managed_default_content_settings.images': 2}
    options.add_experimental_option('prefs', prefs)
    
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(TIMEOUT * 2)  # Double the config timeout for safety
    
    return driver


def scroll_page(driver, max_scrolls=5, pause_time=2):
    """Scroll down to load dynamic content"""
    last_height = driver.execute_script("return document.body.scrollHeight")
    scrolls = 0
    
    for i in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time + random.uniform(0, 1))
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        scrolls += 1
        
        if new_height == last_height:
            break
            
        last_height = new_height
    
    return scrolls


def extract_jobs_from_html(html):
    """Extract job listings from HTML"""
    soup = BeautifulSoup(html, 'lxml')
    jobs = []
    seen = set()
    
    # Get all links
    all_a_tags = soup.find_all('a', href=True)
    
    for a in all_a_tags:
        href = a.get('href', '')
        
        # Filter for job links (contains -jv)
        if not ('-jv' in href):
            continue
        
        # Skip non-job pages
        if any(skip in href for skip in ['login', 'signup', 'profile', 'company', 'muc-luong']):
            continue
        
        # Get title
        title = normalize_spaces(a.get_text())
        
        # Extract from URL if no title
        if len(title) < 5:
            match = re.search(r'/([^/]+)-\d+-jv', href)
            if match:
                title = match.group(1).replace('-', ' ').title()
        
        if len(title) < 5:
            continue
        
        # Build full URL
        if href.startswith('http'):
            link = href
        elif href.startswith('/'):
            link = f"{BASE_URL_VNW}{href}"
        else:
            link = f"{BASE_URL_VNW}/{href}"
        
        link = clean_url(link)
        
        if link in seen:
            continue
        
        seen.add(link)
        jobs.append({
            'tieu_de': title,
            'link': link
        })
    
    return jobs


def scrape_page(driver, url, scroll=True, max_retries=3):
    """
    Scrape a single page with retry mechanism
    
    Args:
        driver: WebDriver instance
        url: URL to scrape
        scroll: Whether to scroll the page
        max_retries: Maximum retry attempts on failure
    
    Returns:
        List of jobs
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  📄 Loading: {url}")
            driver.get(url)
            
            # Wait for content
            time.sleep(5)
            
            # Scroll if needed
            if scroll:
                scrolls = scroll_page(driver, max_scrolls=MAX_SCROLLS, pause_time=SCROLL_PAUSE_TIME)
                print(f"     Scrolled {scrolls} times")
            
            # Extract jobs
            html = driver.page_source
            jobs = extract_jobs_from_html(html)
            
            print(f"     Found {len(jobs)} jobs")
            
            return jobs
            
        except TimeoutException as e:
            print(f"     ⚠️  Timeout on attempt {attempt}/{max_retries}")
            
            if attempt < max_retries:
                # Wait before retry
                wait_time = 5 * attempt
                print(f"     ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                
                # Try to recover driver
                try:
                    driver.refresh()
                except:
                    pass
            else:
                print(f"     ❌ Failed after {max_retries} attempts, skipping page")
                return []  # Return empty list instead of crashing
        
        except Exception as e:
            print(f"     ❌ Error: {type(e).__name__}: {str(e)[:100]}")
            
            if attempt < max_retries:
                time.sleep(3)
            else:
                print(f"     ❌ Failed after {max_retries} attempts, skipping page")
                return []
    
    return []


def scrape_vietnamworks_multi_strategy(max_jobs=None, max_pages=20, headless=None):
    """
    Scrape VietnamWorks using multiple strategies:
    1. Category pages (IT Software)
    2. Pagination (page=1, page=2, ...)
    3. Different job categories
    
    Args:
        max_jobs: Maximum number of jobs to collect (default from config: MAX_JOBS_VNW)
        max_pages: Maximum pages to scrape per category
        headless: Run browser in headless mode (default from config: HEADLESS)
    
    Returns:
        List of job dicts
    """
    # Use config defaults if not specified
    if max_jobs is None:
        max_jobs = MAX_JOBS_VNW
    if headless is None:
        headless = HEADLESS
    
    print(f"🚀 Starting VietnamWorks Multi-Strategy Scraper")
    print(f"Target: {max_jobs} jobs, Max pages per category: {max_pages}")
    
    # Check for checkpoint file
    checkpoint_file = RAW_LIST_PATH_VNW.replace('.csv', '_checkpoint.csv')
    all_jobs = []
    
    try:
        import os
        if os.path.exists(checkpoint_file):
            df_checkpoint = pd.read_csv(checkpoint_file)
            all_jobs = df_checkpoint.to_dict('records')
            print(f"\n📂 Resumed from checkpoint: {len(all_jobs)} jobs already scraped")
    except:
        pass
    
    driver = setup_driver(headless=headless)
    
    try:
        # Strategy 1: IT Software category with pagination
        print(f"\n📚 Strategy 1: IT Software Category")
        base_url = f"{BASE_URL_VNW}/it-phan-mem-kv"
        
        pages_scraped = 0
        for page in range(1, max_pages + 1):
            if len(all_jobs) >= max_jobs:
                break
            
            # Restart driver every 10 pages to prevent memory issues
            if pages_scraped > 0 and pages_scraped % 10 == 0:
                print(f"\n  🔄 Restarting browser (scraped {pages_scraped} pages)...")
                driver.quit()
                time.sleep(2)
                driver = setup_driver(headless=headless)
            
            # VietnamWorks pagination: ?page=N
            url = f"{base_url}?page={page}"
            
            jobs = scrape_page(driver, url, scroll=True, max_retries=3)
            pages_scraped += 1
            
            if not jobs:
                print(f"     No jobs found, stopping pagination")
                break
            
            all_jobs.extend(jobs)
            
            # Deduplicate
            df_tmp = pd.DataFrame(all_jobs).drop_duplicates(subset=['link'])
            all_jobs = df_tmp.to_dict('records')
            
            print(f"  ✅ Total unique jobs: {len(all_jobs)}")
            
            # Save checkpoint every 50 jobs
            if len(all_jobs) % 50 == 0 and len(all_jobs) > 0:
                checkpoint_file = RAW_LIST_PATH_VNW.replace('.csv', '_checkpoint.csv')
                pd.DataFrame(all_jobs).to_csv(checkpoint_file, index=False, encoding='utf-8-sig')
                print(f"  💾 Checkpoint saved: {checkpoint_file}")
            
            # Random delay between pages
            sleep_random(2.0, 4.0)
        
        # Strategy 2: If still need more, try search with pagination
        if len(all_jobs) < max_jobs:
            print(f"\n📚 Strategy 2: Search Query with Pagination")
            base_url = f"{BASE_URL_VNW}/viec-lam?q=it"
            
            for page in range(1, min(10, max_pages) + 1):  # Limit search to 10 pages
                if len(all_jobs) >= max_jobs:
                    break
                
                url = f"{base_url}&page={page}"
                jobs = scrape_page(driver, url, scroll=True)
                
                if not jobs:
                    break
                
                all_jobs.extend(jobs)
                df_tmp = pd.DataFrame(all_jobs).drop_duplicates(subset=['link'])
                all_jobs = df_tmp.to_dict('records')
                
                print(f"  ✅ Total unique jobs: {len(all_jobs)}")
                sleep_random(2.0, 4.0)
        
        # Strategy 3: Try other IT categories
        if len(all_jobs) < max_jobs:
            print(f"\n📚 Strategy 3: Other IT Categories")
            
            categories = [
                f"{BASE_URL_VNW}/viec-lam?g=5",  # IT general (g=5)
                f"{BASE_URL_VNW}/data-analyst-kv",  # Data
                f"{BASE_URL_VNW}/mobile-developer-kv",  # Mobile
            ]
            
            for cat_url in categories:
                if len(all_jobs) >= max_jobs:
                    break
                
                print(f"\n  Category: {cat_url}")
                
                for page in range(1, min(5, max_pages) + 1):  # Max 5 pages per category
                    if len(all_jobs) >= max_jobs:
                        break
                    
                    url = f"{cat_url}&page={page}" if '?' in cat_url else f"{cat_url}?page={page}"
                    jobs = scrape_page(driver, url, scroll=False)  # No scroll for speed
                    
                    if not jobs:
                        break
                    
                    all_jobs.extend(jobs)
                    df_tmp = pd.DataFrame(all_jobs).drop_duplicates(subset=['link'])
                    all_jobs = df_tmp.to_dict('records')
                    
                    print(f"  ✅ Total unique jobs: {len(all_jobs)}")
                    sleep_random(2.0, 4.0)
    
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.quit()
        print("\n🔒 Browser closed")
    
    return all_jobs[:max_jobs]


if __name__ == "__main__":
    import os
    os.makedirs(os.path.dirname(RAW_LIST_PATH_VNW), exist_ok=True)
    
    # Use defaults from config_vietnamworks.py
    # MAX_JOBS_VNW, HEADLESS from config
    jobs = scrape_vietnamworks_multi_strategy()
    
    df = pd.DataFrame(jobs)
    df.to_csv(RAW_LIST_PATH_VNW, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 Scraping completed!")
    print(f"✅ Saved: {RAW_LIST_PATH_VNW}")
    print(f"📊 Total jobs: {len(df)}")
    
    if len(df) > 0:
        print(f"\n📋 First 5 jobs:")
        print(df.head(5).to_string(index=False))
        
        print(f"\n📋 Last 5 jobs:")
        print(df.tail(5).to_string(index=False))