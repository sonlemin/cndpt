# src/02_scrape_detail_vietnamworks.py
"""
VietnamWorks Detail Scraper using Selenium

Scrapes job detail pages from VietnamWorks.
Uses Selenium because VietnamWorks uses client-side rendering.

Features:
- Resume capability (checkpoint saving)
- Extracts full job description
- Handles dynamic content loading
"""

import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

from config_vietnamworks import (
    RAW_LIST_PATH_VNW,
    RAW_DETAIL_PATH_VNW,
    CHECKPOINT_INTERVAL,
)
from utils import clean_url, sleep_random


def setup_driver(headless=True):
    """Setup Chrome WebDriver"""
    options = webdriver.ChromeOptions()
    
    if headless:
        options.add_argument('--headless')
    
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36')
    
    # Disable images to speed up
    prefs = {'profile.managed_default_content_settings.images': 2}
    options.add_experimental_option('prefs', prefs)
    
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)
    
    return driver


def extract_job_content(driver):
    """
    Extract job description from detail page
    
    VietnamWorks structure:
    - Job description is typically in a main content div
    - May need to wait for dynamic loading
    
    Args:
        driver: WebDriver instance
    
    Returns:
        Text content of job description
    """
    try:
        # Wait for main content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body'))
        )
        
        # Get page source
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        
        # Extract text from body
        # Remove script and style tags
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav']):
            tag.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Normalize spaces
        text = ' '.join(text.split())
        
        return text
        
    except Exception as e:
        print(f"  ⚠️  Error extracting content: {e}")
        return ""


def scrape_detail(headless=True):
    """
    Scrape job details from VietnamWorks
    
    Args:
        headless: Run browser in headless mode
    """
    # Read job list
    df = pd.read_csv(RAW_LIST_PATH_VNW)
    df['link'] = df['link'].map(clean_url)
    
    # Resume from checkpoint if exists
    try:
        done = pd.read_csv(RAW_DETAIL_PATH_VNW)
        rows = done.to_dict('records')
        done_links = set(done['link'].astype(str))
        print(f"📂 Resuming from checkpoint: {len(done_links)} jobs already scraped")
    except:
        rows, done_links = [], set()
        print("🆕 Starting fresh scrape")
    
    # Setup driver
    driver = setup_driver(headless=headless)
    
    try:
        for i, (title, link) in enumerate(zip(df['tieu_de'], df['link']), start=1):
            if link in done_links:
                continue
            
            print(f"[{len(done_links)+1}/{len(df)}] {title[:60]}...")
            
            try:
                # Load job detail page
                driver.get(link)
                
                # Wait a bit for content to load
                time.sleep(3)
                
                # Extract content
                content = extract_job_content(driver)
                
                rows.append({
                    'tieu_de': title,
                    'link': link,
                    'noi_dung': content
                })
                done_links.add(link)
                
                # Save checkpoint every N jobs
                if len(rows) % CHECKPOINT_INTERVAL == 0:
                    pd.DataFrame(rows).to_csv(RAW_DETAIL_PATH_VNW, index=False, encoding='utf-8-sig')
                    print(f"  💾 Checkpoint saved: {len(rows)} jobs")
                
                # Random delay
                sleep_random(2.0, 4.0)
                
            except Exception as e:
                print(f"  ❌ Error scraping {link}: {e}")
                continue
    
    finally:
        driver.quit()
        print("\n🔒 Browser closed")
    
    # Final save
    pd.DataFrame(rows).to_csv(RAW_DETAIL_PATH_VNW, index=False, encoding='utf-8-sig')
    print(f"\n✅ Scraping completed!")
    print(f"📁 Saved: {RAW_DETAIL_PATH_VNW}")
    print(f"📊 Total jobs: {len(rows)}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(RAW_DETAIL_PATH_VNW), exist_ok=True)
    
    scrape_detail(headless=True)  # Set to False to see browser