# Collect Malayalam text from web

import requests
from bs4 import BeautifulSoup
import json
import time

# List of websites to scrape
MALAYALAM_SITES = [
    'https://www.mathrubhumi.com/',
    'https://www.manoramaonline.com/',
    'https://www.emalayalee.com/',
    'https://www.asianetnews.com/',
    'https://malayalam.samayam.com/latest-news/articlelist/48237651.cms',
    'https://www.keralakaumudi.com/',
    'https://www.madhyamam.com/',
]

def get_malayalam_text(url):
    # Fetch page content
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find text elements
        paragraphs = soup.find_all(['p', 'div', 'article', 'section'])
        
        text_data = []
        for elem in paragraphs:
            text = elem.get_text(separator=' ').strip()
            # Check for Malayalam content
            if len(text) > 50 and any('\u0D00' <= c <= '\u0D7F' for c in text):
                text_data.append(text)
        
        return text_data
    
    except Exception as e:
        print(f"error: {e}")
        return []

def save_collected_data(data, filename='data/malayalam_corpus.json'):
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"saved {len(data)} texts")

def main():
    print("starting collection...")
    
    all_texts = []
    
    # Scrape each site
    for site in MALAYALAM_SITES:
        print(f"scraping {site}")
        texts = get_malayalam_text(site)
        all_texts.extend(texts)
        time.sleep(2)
    
    print(f"total collected: {len(all_texts)}")
    
    # Save collected data
    save_collected_data(all_texts)
    
    print("done")

if __name__ == '__main__':
    main()
