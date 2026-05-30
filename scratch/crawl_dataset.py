import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys
import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# Set standard output encoding to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def crawl_tingia_fake_news():
    print("=== STARTING TINGIA.GOV.VN CRAWLER ===")
    all_articles = []
    page = 1
    
    while True:
        url = f"https://tingia.gov.vn/tin-vua-check/{page}/" if page > 1 else "https://tingia.gov.vn/tin-vua-check/"
        print(f"Crawling page {page}: {url}")
        
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                print(f"Page {page} returned status {response.status_code}. Stopping crawl.")
                break
                
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Select article links
            links = []
            selectors = [
                'div.tc-post-grid-style9 h2.title a',
                'div.tc-post-grid-style9 h6.title a'
            ]
            for sel in selectors:
                for item in soup.select(sel):
                    href = item.get('href', '')
                    if href:
                        if not href.startswith('http'):
                            href = "https://tingia.gov.vn" + href
                        links.append(href)
            
            # Remove duplicates
            links = list(set(links))
            if not links:
                print("No article links found on this page. Stopping.")
                break
                
            print(f"Found {len(links)} unique links on page {page}.")
            
            # Crawl detail of each article
            for link in links:
                print(f"  Fetching: {link}")
                try:
                    res = requests.get(link, headers=HEADERS, timeout=10)
                    if res.status_code != 200:
                        continue
                    
                    art_soup = BeautifulSoup(res.content, "html.parser")
                    
                    title_elem = art_soup.select_one("h1.post-title.entry-title")
                    title = title_elem.get_text(strip=True) if title_elem else ""
                    
                    sapo_elem = art_soup.select_one("div.content-detail-sapo")
                    sapo = sapo_elem.get_text(strip=True) if sapo_elem else ""
                    
                    content_elem = art_soup.select_one("div#maincontent")
                    if content_elem:
                        paras = [p.get_text(strip=True) for p in content_elem.find_all("p") if p.get_text(strip=True)]
                        content = "\n".join(paras)
                    else:
                        content = ""
                        
                    if title or content:
                        # Combine sapo and content for a single text body
                        full_text = f"{title}. {sapo}. {content}".strip()
                        all_articles.append({
                            "post_message": full_text,
                            "label": 1  # Fake News
                        })
                    time.sleep(0.1) # Gentle request rate limit
                except Exception as e:
                    print(f"  Error crawling detail {link}: {e}")
                    
            page += 1
            # Limit to page 50 to get as much as possible
            if page > 50:
                print("Reached page limit (page 50). Stopping.")
                break
                
        except Exception as e:
            print(f"Error crawling page {page}: {e}")
            break
            
    print(f"=== Crawled {len(all_articles)} fake news items from tingia.gov.vn ===\n")
    return all_articles

def crawl_vnexpress_real_news(target_count):
    print("=== STARTING VNEXPRESS.NET CRAWLER ===")
    rss_urls = [
        "https://vnexpress.net/rss/thoi-su.rss",
        "https://vnexpress.net/rss/the-gioi.rss",
        "https://vnexpress.net/rss/kinh-doanh.rss",
        "https://vnexpress.net/rss/startup.rss",
        "https://vnexpress.net/rss/giai-tri.rss",
        "https://vnexpress.net/rss/the-thao.rss",
        "https://vnexpress.net/rss/phap-luat.rss",
        "https://vnexpress.net/rss/giao-duc.rss",
        "https://vnexpress.net/rss/suc-khoe.rss",
        "https://vnexpress.net/rss/doi-song.rss",
        "https://vnexpress.net/rss/du-lich.rss",
        "https://vnexpress.net/rss/khoa-hoc.rss",
        "https://vnexpress.net/rss/so-hoa.rss",
        "https://vnexpress.net/rss/oto-xe-may.rss"
    ]
    
    article_links = []
    for rss in rss_urls:
        print(f"Fetching RSS: {rss}")
        try:
            res = requests.get(rss, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(res.content, "xml") # Parse as XML
            for item in soup.find_all("item"):
                link = item.find("link")
                if link and link.text:
                    article_links.append(link.text.strip())
        except Exception as e:
            print(f"Error reading RSS {rss}: {e}")
            
    article_links = list(set(article_links))
    print(f"Found {len(article_links)} unique real news links from RSS feeds.")
    
    real_articles = []
    for link in article_links:
        if len(real_articles) >= target_count:
            print(f"Reached target real news count ({target_count}). Stopping.")
            break
            
        print(f"  Fetching: {link}")
        try:
            res = requests.get(link, headers=HEADERS, timeout=10)
            if res.status_code != 200:
                continue
                
            soup = BeautifulSoup(res.content, "html.parser")
            
            title_elem = soup.find("h1", class_="title-detail")
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            sapo_elem = soup.find("p", class_="description")
            sapo = sapo_elem.get_text(strip=True) if sapo_elem else ""
            
            # Extract content from div.fck_detail p.Normal
            content_elem = soup.find("div", class_="fck_detail")
            if content_elem:
                paras = [p.get_text(strip=True) for p in content_elem.find_all("p", class_="Normal") if p.get_text(strip=True)]
                content = "\n".join(paras)
            else:
                content = ""
                
            if title or content:
                full_text = f"{title}. {sapo}. {content}".strip()
                real_articles.append({
                    "post_message": full_text,
                    "label": 0  # Real News
                })
            time.sleep(0.05)
        except Exception as e:
            print(f"  Error fetching VnExpress article {link}: {e}")
            
    print(f"=== Crawled {len(real_articles)} real news items from VnExpress ===\n")
    return real_articles

if __name__ == "__main__":
    # 1. Crawl verified fake news
    fake_news = crawl_tingia_fake_news()
    
    if not fake_news:
        print("No fake news crawled. Exiting.")
        sys.exit(1)
        
    # 2. Crawl 10,000 real news articles
    target_real_count = 10000
    real_news = crawl_vnexpress_real_news(target_real_count)
    
    # 3. Combine and save to data/tingia_crawled.csv
    combined_data = fake_news + real_news
    df = pd.DataFrame(combined_data)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/tingia_crawled.csv", index=False)
    print(f"Saved {len(df)} balanced rows to data/tingia_crawled.csv.")
    
    # 4. Split into Train (80%), Val (10%), Test (10%)
    if len(df) >= 10:
        train_df, temp_df = train_test_split(
            df, test_size=0.20, random_state=42, stratify=df['label']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
        )
        
        train_df.to_csv("data/tingia_train.csv", index=False)
        val_df.to_csv("data/tingia_val.csv", index=False)
        test_df.to_csv("data/tingia_test.csv", index=False)
        
        print("Split tingia dataset successfully:")
        print(f"  -> Train: {len(train_df)} rows")
        print(f"  -> Val:   {len(val_df)} rows")
        print(f"  -> Test:  {len(test_df)} rows")
    else:
        print("Dataset too small to split.")
