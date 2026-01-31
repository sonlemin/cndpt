#!/usr/bin/env python3
# test_selenium_setup.py
"""
Test Selenium Setup

Quick test to verify Selenium and ChromeDriver are working correctly
before running the full VietnamWorks scraper.

Run this first: python test_selenium_setup.py
"""

import sys


def test_selenium_import():
    """Test if selenium is installed"""
    print("=" * 60)
    print("TEST 1: Selenium Import")
    print("=" * 60)
    
    try:
        import selenium
        print(f"✅ Selenium installed: version {selenium.__version__}")
        return True
    except ImportError:
        print("❌ Selenium not installed")
        print("   Install: pip install selenium")
        return False


def test_chromedriver():
    """Test if ChromeDriver is available"""
    print("\n" + "=" * 60)
    print("TEST 2: ChromeDriver Availability")
    print("=" * 60)
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        # Setup headless Chrome
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        print("🔧 Attempting to start ChromeDriver...")
        driver = webdriver.Chrome(options=options)
        
        print("✅ ChromeDriver started successfully")
        
        # Get version info
        caps = driver.capabilities
        browser_version = caps.get('browserVersion', 'unknown')
        chromedriver_version = caps.get('chrome', {}).get('chromedriverVersion', 'unknown')
        
        print(f"   Chrome version: {browser_version}")
        print(f"   ChromeDriver version: {chromedriver_version}")
        
        driver.quit()
        return True
        
    except Exception as e:
        print(f"❌ ChromeDriver failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Chrome is installed")
        print("2. Update selenium: pip install --upgrade selenium")
        print("3. If using selenium < 4.6, download ChromeDriver manually:")
        print("   https://googlechromelabs.github.io/chrome-for-testing/")
        return False


def test_vietnamworks_access():
    """Test if we can access VietnamWorks"""
    print("\n" + "=" * 60)
    print("TEST 3: VietnamWorks Access")
    print("=" * 60)
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import time
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        print("🌐 Loading VietnamWorks IT jobs page...")
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        
        url = "https://www.vietnamworks.com/viec-lam?q=it&sorting=relevant"
        driver.get(url)
        
        # Wait a bit for JavaScript to load
        time.sleep(5)
        
        # Check if page loaded
        page_source = driver.page_source
        
        if len(page_source) > 1000:
            print(f"✅ Page loaded successfully ({len(page_source):,} bytes)")
        else:
            print(f"⚠️  Page seems too small ({len(page_source)} bytes)")
        
        # Try to find job links
        try:
            job_links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="-jv"]')
            print(f"✅ Found {len(job_links)} job links")
            
            if len(job_links) > 0:
                print(f"   Example: {job_links[0].get_attribute('href')}")
        except:
            print("⚠️  Could not find job links (might be OK if page structure changed)")
        
        driver.quit()
        return True
        
    except Exception as e:
        print(f"❌ Access failed: {e}")
        return False


def test_dependencies():
    """Test other dependencies"""
    print("\n" + "=" * 60)
    print("TEST 4: Other Dependencies")
    print("=" * 60)
    
    deps = {
        'pandas': 'pandas',
        'beautifulsoup4': 'bs4',
        'lxml': 'lxml',
        'requests': 'requests'
    }
    
    all_ok = True
    
    for package_name, import_name in deps.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Install: pip install {package_name}")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("VIETNAMWORKS SCRAPER - SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Selenium Import", test_selenium_import()))
    results.append(("ChromeDriver", test_chromedriver()))
    results.append(("VietnamWorks Access", test_vietnamworks_access()))
    results.append(("Dependencies", test_dependencies()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! You're ready to scrape VietnamWorks!")
        print("\nNext steps:")
        print("1. python src/01_scrape_list_vietnamworks.py")
        print("2. python src/02_scrape_detail_vietnamworks.py")
        print("=" * 60)
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())