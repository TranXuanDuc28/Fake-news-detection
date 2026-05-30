import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys

# Set standard output encoding to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def crawl_page_listings(page_num):
    url = f"https://tingia.gov.vn/tin-vua-check/{page_num}/" if page_num > 1 else "https://tingia.gov.vn/tin-vua-check/"
    print(f"-> Crawling listings on page {page_num}: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Error {response.status_code} loading page {page_num}")
            return []
        
        soup = BeautifulSoup(response.content, "html.parser")
        links = []
        
        # Selectors matching listing links
        selectors = [
            'div.tc-post-grid-style9 h2.title a',
            'div.tc-post-grid-style9 h6.title a'
        ]
        
        for sel in selectors:
            for item in soup.select(sel):
                title = item.get_text(strip=True)
                href = item.get('href', '')
                if href:
                    if not href.startswith('http'):
                        href = "https://tingia.gov.vn" + href
                    links.append({"title": title, "url": href})
        
        # Remove duplicate links
        unique_links = []
        seen = set()
        for item in links:
            if item['url'] not in seen:
                seen.add(item['url'])
                unique_links.append(item)
                
        return unique_links
    except Exception as e:
        print(f"Exception while crawling page {page_num}: {e}")
        return []

def crawl_article_detail(url):
    print(f"  -> Fetching article: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Select title
        title_elem = soup.select_one("h1.post-title.entry-title")
        title = title_elem.get_text(strip=True) if title_elem else ""
        
        # Select Sapo / Summary
        sapo_elem = soup.select_one("div.content-detail-sapo")
        sapo = sapo_elem.get_text(strip=True) if sapo_elem else ""
        
        # Select main body content
        content_elem = soup.select_one("div#maincontent")
        if content_elem:
            paragraphs = [p.get_text(strip=True) for p in content_elem.find_all("p") if p.get_text(strip=True)]
            body = "\n".join(paragraphs)
        else:
            body = ""
            
        return {
            "title": title,
            "sapo": sapo,
            "content": body,
            "url": url
        }
    except Exception as e:
        print(f"  Error fetching article {url}: {e}")
        return None

if __name__ == "__main__":
    print("Testing crawler on page 1 of tingia.gov.vn...")
    links = crawl_page_listings(1)
    print(f"Found {len(links)} links on page 1.")
    
    # Try fetching details of the first 3 links as a test
    for i, item in enumerate(links[:3]):
        print(f"\nArticle #{i+1}:")
        detail = crawl_article_detail(item['url'])
        if detail:
            print("Title:", detail['title'])
            print("Sapo:", detail['sapo'])
            print("Content Snippet:", detail['content'][:150] + "...")
