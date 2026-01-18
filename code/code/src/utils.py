# src/utils.py
import time, random, re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlsplit, urlunsplit

def clean_url(u: str) -> str:
    """Bỏ query/utm để tránh trùng link"""
    parts = urlsplit(str(u))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

def html_to_text(html: str) -> str:
    """Chuyển HTML -> text thô"""
    soup = BeautifulSoup(html, "lxml")
    return " ".join(soup.get_text(" ", strip=True).split())

def fetch_html(session: requests.Session, url: str, headers=None, retry=2, timeout=20) -> str:
    for _ in range(retry):
        r = session.get(url, headers=headers, timeout=timeout)
        if r.status_code == 429:
            time.sleep(random.uniform(40, 70))  # bị chặn
            continue
        r.raise_for_status()
        return r.text
    return ""

def sleep_random(a=1.0, b=2.0):
    time.sleep(random.uniform(a, b))

def normalize_spaces(s: str) -> str:
    return " ".join(str(s).strip().split())

def clean_text_vn(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)                 # bỏ link
    text = re.sub(r"[^\wÀ-ỹ\s\+\#\.\-]", " ", text)      # giữ chữ VN + ký tự kỹ thuật
    text = re.sub(r"\s+", " ", text).strip()
    return text
"""
utils.py

Các hàm tiện ích dùng chung cho toàn bộ project.

Bao gồm:
- Network utilities: fetch_html, sleep_random
- Text processing: clean_text, normalize_spaces, html_to_text
- URL handling: clean_url
- Data validation: validate_dataframe
"""

import re
import time
import random
from urllib.parse import urlsplit, urlunsplit
from typing import Optional, Tuple
import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError
from bs4 import BeautifulSoup
import pandas as pd


def clean_url(url: str) -> str:
    """
    Loại bỏ query parameters và fragments khỏi URL.
    
    TẠI SAO:
    - URL có query params khác nhau nhưng trỏ về cùng 1 page
    - VD: /job-123?utm_source=facebook vs /job-123?utm_source=google
    - Cần chuẩn hóa để tránh duplicate
    
    SỬ DỤNG Ở ĐÂU:
    - 03_preprocess_clean.py: Deduplicate links
    
    VÍ DỤ:
        Input:  "https://topcv.vn/job-123?utm_source=fb&ref=home#section1"
        Output: "https://topcv.vn/job-123"
    
    Args:
        url: URL cần clean
    
    Returns:
        URL đã loại bỏ query params và fragments
    """
    parts = urlsplit(url)
    # Keep scheme, netloc, path only
    # Drop query and fragment
    clean_parts = (parts.scheme, parts.netloc, parts.path, '', '')
    return urlunsplit(clean_parts)


def normalize_spaces(text: str) -> str:
    """
    Chuẩn hóa whitespace: loại bỏ khoảng trắng thừa.
    
    TẠI SAO:
    - HTML thường có nhiều spaces, tabs, newlines thừa
    - "python    django" và "python django" nên được coi là giống nhau
    
    SỬ DỤNG Ở ĐÂU:
    - html_to_text(): Sau khi extract text
    - clean_text_vn(): Trong quá trình clean
    
    VÍ DỤ:
        Input:  "python    django\n\npostgresql"
        Output: "python django postgresql"
    
    Args:
        text: Text cần normalize
    
    Returns:
        Text với single spaces
    """
    # Split by any whitespace, then join with single space
    return ' '.join(text.split())


def html_to_text(html: str) -> str:
    """
    Convert HTML sang plain text.
    
    TẠI SAO:
    - Cần extract text content từ HTML response
    - Loại bỏ tags, scripts, styles
    
    CÁCH HOẠT ĐỘNG:
    1. Parse HTML với BeautifulSoup
    2. Remove <script> và <style> tags
    3. Extract text
    4. Normalize spaces
    
    SỬ DỤNG Ở ĐÂU:
    - 02_scrape_detail_topcv.py: Parse job content
    
    VÍ DỤ:
        Input:  "<div>Job: <b>Python</b> Developer</div><script>alert('hi')</script>"
        Output: "Job: Python Developer"
    
    Args:
        html: HTML string
    
    Returns:
        Plain text content
    """
    try:
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove script and style elements
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Normalize spaces
        text = normalize_spaces(text)
        
        return text
    except Exception as e:
        print(f"⚠️ Error parsing HTML: {e}")
        return ""


def fetch_html(
    session: requests.Session,
    url: str,
    headers: Optional[dict] = None,
    retry: int = 3,
    timeout: int = 30
) -> Optional[str]:
    """
    Fetch HTML từ URL với retry mechanism.
    
    TẠI SAO CẦN RETRY:
    - Network không ổn định (timeout, connection reset)
    - Server overload (503)
    - Rate limiting (429)
    
    CHIẾN LƯỢC RETRY:
    - Timeout/ConnectionError: Retry ngay với exponential backoff
    - 429 (Rate limit): Sleep 40-70s rồi retry
    - 404/other errors: Không retry (vô ích)
    
    SỬ DỤNG Ở ĐÂU:
    - 01_scrape_list_topcv.py: Fetch search result pages
    - 02_scrape_detail_topcv.py: Fetch job detail pages
    
    VÍ DỤ:
        session = requests.Session()
        html = fetch_html(session, "https://topcv.vn/job-123")
        if html:
            content = html_to_text(html)
    
    Args:
        session: requests.Session object (for connection reuse)
        url: URL to fetch
        headers: Optional headers dict
        retry: Number of retry attempts (default: 3)
        timeout: Timeout in seconds (default: 30)
    
    Returns:
        HTML string if success, None if failed
    """
    for attempt in range(1, retry + 1):
        try:
            response = session.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                if attempt < retry:
                    sleep_time = random.uniform(40, 70)
                    print(f"⚠️ Rate limited (429). Sleeping {sleep_time:.0f}s...")
                    time.sleep(sleep_time)
                    continue
                else:
                    print(f"❌ Rate limited after {retry} retries")
                    return None
            
            # Raise for bad status codes
            response.raise_for_status()
            
            # Success
            return response.text
            
        except Timeout:
            if attempt < retry:
                sleep_time = 5 * attempt  # Exponential backoff
                print(f"⏱️  Timeout (attempt {attempt}/{retry}). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            else:
                print(f"❌ Timeout after {retry} retries")
                return None
                
        except ConnectionError:
            if attempt < retry:
                sleep_time = 5 * attempt
                print(f"🔌 Connection error (attempt {attempt}/{retry}). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            else:
                print(f"❌ Connection error after {retry} retries")
                return None
                
        except HTTPError as e:
            # Don't retry for 404, 403, etc
            print(f"❌ HTTP error {e.response.status_code}: {url}")
            return None
            
        except Exception as e:
            print(f"❌ Unexpected error: {type(e).__name__}: {e}")
            return None
    
    return None


def sleep_random(min_seconds: float, max_seconds: float) -> None:
    """
    Sleep random time trong khoảng [min, max].
    
    TẠI SAO CẦN RANDOM:
    - Fixed sleep pattern → dễ bị detect là bot
    - Random pattern → giống human behavior hơn
    
    SỬ DỤNG Ở ĐÂU:
    - 01_scrape_list_topcv.py: Giữa các pages
    - 02_scrape_detail_topcv.py: Giữa các jobs
    
    VÍ DỤ:
        sleep_random(2.0, 5.0)  # Sleep 2-5 giây
    
    Args:
        min_seconds: Minimum sleep time
        max_seconds: Maximum sleep time
    """
    sleep_time = random.uniform(min_seconds, max_seconds)
    time.sleep(sleep_time)


def clean_text_vn(text: str) -> str:
    """
    Clean text tiếng Việt: lowercase, loại bỏ URLs, emails, special chars.
    
    TẠI SAO:
    - Chuẩn hóa để dễ so sánh và search
    - Loại bỏ noise (URLs, emails không cần thiết)
    - Preserve technical terms (C++, C#, .NET)
    
    CÁCH HOẠT ĐỘNG:
    1. Lowercase
    2. Preserve C++, C#, .NET (technical terms)
    3. Remove URLs
    4. Remove emails
    5. Normalize spaces
    
    SỬ DỤNG Ở ĐÂU:
    - 03_preprocess_clean.py: Clean title và content
    
    VÍ DỤ:
        Input:  "Senior C++ Developer\nEmail: hr@company.com\nhttps://company.com"
        Output: "senior c++ developer"
    
    Args:
        text: Text cần clean
    
    Returns:
        Cleaned text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Preserve technical terms before cleaning
    # Replace với placeholders
    text = text.replace('c++', '__CPP__')
    text = text.replace('c#', '__CSHARP__')
    text = text.replace('.net', '__DOTNET__')
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters (keep alphanumeric, Vietnamese chars, spaces)
    # Keep: a-z, 0-9, Vietnamese chars, spaces, hyphens
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\-]', ' ', text)
    
    # Restore technical terms
    text = text.replace('__cpp__', 'c++')
    text = text.replace('__csharp__', 'c#')
    text = text.replace('__dotnet__', '.net')
    
    # Normalize spaces
    text = normalize_spaces(text)
    
    return text


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    name: str = "DataFrame"
) -> None:
    """
    Validate DataFrame có đủ columns cần thiết.
    
    TẠI SAO:
    - Early detection of data issues
    - Clear error messages
    - Prevent cryptic errors downstream
    
    SỬ DỤNG Ở ĐÂU:
    - Đầu mỗi script: Validate input data
    
    VÍ DỤ:
        df = pd.read_csv('data.csv')
        validate_dataframe(df, ['title', 'link', 'content'], 'Job data')
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for error messages
    
    Raises:
        ValueError: If missing required columns
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} thiếu columns: {missing}\n"
            f"Có: {list(df.columns)}\n"
            f"Cần: {required_columns}"
        )


def extract_salary_range(text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extract salary range từ text (VND hoặc USD).
    
    PATTERNS SUPPORTED:
    1. "15-25 triệu" → (15.0, 25.0, 20.0)
    2. "20 triệu" → (20.0, 20.0, 20.0)
    3. "1000-1500 USD" → (23.0, 34.5, 28.75) - convert to VND triệu
    4. "thỏa thuận" → (None, None, None)
    
    CONVERSION RATE:
    - 1 USD ≈ 23,000 VND = 0.023 triệu VND
    
    SỬ DỤNG Ở ĐÂU:
    - 04_extract_features.py: Extract salary from job descriptions
    
    VÍ DỤ:
        text = "Lương: 15-25 triệu VND"
        min_sal, max_sal, avg_sal = extract_salary_range(text)
        # → (15.0, 25.0, 20.0)
    
    Args:
        text: Text chứa salary info
    
    Returns:
        Tuple of (min_salary, max_salary, avg_salary) in triệu VND
        Returns (None, None, None) if not found or "thỏa thuận"
    """
    if pd.isna(text) or not isinstance(text, str):
        return None, None, None
    
    text = text.lower()
    
    # Check for "thỏa thuận" / "협의" / "negotiable"
    if re.search(r'thỏa thuận|thoả thuận|協議|negotiable|competitive', text):
        return None, None, None
    
    # Try USD first (convert to triệu VND)
    # Pattern: "1000-1500 USD" hoặc "1000 USD"
    usd_pattern = r'(\d+)\s*[-~]\s*(\d+)\s*(?:usd|\$)'
    match = re.search(usd_pattern, text)
    if match:
        min_usd = float(match.group(1))
        max_usd = float(match.group(2))
        # 1 USD ≈ 23,000 VND = 0.023 triệu VND
        min_vnd = min_usd * 0.023
        max_vnd = max_usd * 0.023
        avg_vnd = (min_vnd + max_vnd) / 2
        return min_vnd, max_vnd, avg_vnd
    
    # Single USD value
    usd_single = r'(\d+)\s*(?:usd|\$)'
    match = re.search(usd_single, text)
    if match:
        usd = float(match.group(1))
        vnd = usd * 0.023
        return vnd, vnd, vnd
    
    # VND range: "15-25 triệu" or "15~25tr"
    vnd_range = r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*(?:triệu|tr|trieu|million)'
    match = re.search(vnd_range, text)
    if match:
        min_sal = float(match.group(1))
        max_sal = float(match.group(2))
        avg_sal = (min_sal + max_sal) / 2
        return min_sal, max_sal, avg_sal
    
    # Single VND value: "20 triệu"
    vnd_single = r'(\d+(?:\.\d+)?)\s*(?:triệu|tr|trieu|million)'
    match = re.search(vnd_single, text)
    if match:
        sal = float(match.group(1))
        return sal, sal, sal
    
    return None, None, None


def extract_experience_years(text: str) -> Optional[float]:
    """
    Extract số năm kinh nghiệm yêu cầu từ text.
    
    PATTERNS SUPPORTED:
    1. "3 năm kinh nghiệm" → 3.0
    2. "kinh nghiệm 2-3 năm" → 2.5 (average)
    3. "fresher" / "không yêu cầu" → 0.0
    4. "5+ years experience" → 5.0
    
    SỬ DỤNG Ở ĐÂU:
    - 04_extract_features.py: Extract experience requirement
    
    VÍ DỤ:
        text = "Yêu cầu: 3 năm kinh nghiệm Python"
        years = extract_experience_years(text)
        # → 3.0
    
    Args:
        text: Text chứa experience info
    
    Returns:
        Số năm kinh nghiệm (float), None nếu không tìm thấy
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    text = text.lower()
    
    # Check for fresher / no experience required
    if re.search(r'fresher|không yêu cầu kinh nghiệm|no experience|entry level', text):
        return 0.0
    
    # Pattern 1: "X năm" or "X years"
    pattern1 = r'(\d+)\s*(?:năm|years?|yr|yrs)'
    matches = re.findall(pattern1, text)
    if matches:
        # Take the first occurrence
        return float(matches[0])
    
    # Pattern 2: Range "X-Y năm" → average
    pattern2 = r'(\d+)\s*[-~]\s*(\d+)\s*(?:năm|years?)'
    match = re.search(pattern2, text)
    if match:
        min_exp = float(match.group(1))
        max_exp = float(match.group(2))
        return (min_exp + max_exp) / 2
    
    # Pattern 3: "X+ năm"
    pattern3 = r'(\d+)\+\s*(?:năm|years?)'
    match = re.search(pattern3, text)
    if match:
        return float(match.group(1))
    
    return None


def save_checkpoint(data, filepath: str, mode: str = 'dataframe') -> None:
    """
    Save checkpoint để tránh mất data khi crash.
    
    TẠI SAO:
    - Scraping lâu (2-3 giờ) → rủi ro crash cao
    - Checkpoint mỗi 10 items → mất tối đa 10 items khi crash
    
    SỬ DỤNG Ở ĐÂU:
    - 01_scrape_list_topcv.py: Save mỗi 10 pages
    - 02_scrape_detail_topcv.py: Save mỗi 10 jobs
    
    VÍ DỤ:
        rows = []
        for i, item in enumerate(items):
            rows.append(process(item))
            if (i + 1) % 10 == 0:
                save_checkpoint(rows, 'data.csv')
    
    Args:
        data: Data to save (DataFrame or list of dicts)
        filepath: Path to save file
        mode: 'dataframe' or 'list'
    """
    try:
        if mode == 'dataframe':
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False, encoding='utf-8-sig')
            else:
                pd.DataFrame(data).to_csv(filepath, index=False, encoding='utf-8-sig')
        elif mode == 'list':
            pd.DataFrame(data).to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"💾 Checkpoint saved: {len(data)} rows")
    except Exception as e:
        print(f"⚠️ Error saving checkpoint: {e}")


def format_duration(seconds: float) -> str:
    """
    Format duration từ seconds sang human-readable string.
    
    VÍ DỤ:
        format_duration(3665) → "1h 1m 5s"
        format_duration(125) → "2m 5s"
        format_duration(45) → "45s"
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:  # Always show seconds if no other units
        parts.append(f"{secs}s")
    
    return " ".join(parts)


if __name__ == "__main__":
    # Test functions
    print("Testing utils functions...")
    
    # Test clean_url
    url = "https://topcv.vn/job-123?utm_source=fb&ref=home#section"
    print(f"\nclean_url:")
    print(f"  Input:  {url}")
    print(f"  Output: {clean_url(url)}")
    
    # Test normalize_spaces
    text = "python    django\n\npostgresql"
    print(f"\nnormalize_spaces:")
    print(f"  Input:  {repr(text)}")
    print(f"  Output: {repr(normalize_spaces(text))}")
    
    # Test clean_text_vn
    text = "Senior C++ Developer\nEmail: hr@company.com"
    print(f"\nclean_text_vn:")
    print(f"  Input:  {text}")
    print(f"  Output: {clean_text_vn(text)}")
    
    # Test extract_salary_range
    texts = [
        "Lương: 15-25 triệu VND",
        "Salary: 1000-1500 USD",
        "Lương thỏa thuận",
        "20 triệu"
    ]
    print(f"\nextract_salary_range:")
    for t in texts:
        result = extract_salary_range(t)
        print(f"  {t:30s} → {result}")
    
    # Test extract_experience_years
    texts = [
        "Yêu cầu 3 năm kinh nghiệm",
        "Fresher welcome",
        "5+ years of experience",
        "2-3 năm"
    ]
    print(f"\nextract_experience_years:")
    for t in texts:
        result = extract_experience_years(t)
        print(f"  {t:30s} → {result}")
    
    # Test format_duration
    durations = [45, 125, 3665, 7325]
    print(f"\nformat_duration:")
    for d in durations:
        print(f"  {d:5d}s → {format_duration(d)}")
    
    print("\n✅ All tests completed!")