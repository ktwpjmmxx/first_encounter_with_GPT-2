from duckduckgo_search import DDGS
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
import random
import urllib3
import time
import difflib # 類似度判定用

# SSL警告を無視
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ▼▼▼ TAVILYのAPIキーを入力してください ▼▼▼
TAVILY_API_KEY = "YOUR_API_KEY"
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

def get_page_metadata(url):
    """URLからOGP画像を取得"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=3, verify=False)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        og_image = soup.find("meta", property="og:image")
        image_url = og_image["content"] if og_image else None
        return image_url
    except:
        return None

def is_similar(title1, title2):
    """タイトル類似度判定 (60%以上で重複)"""
    if not title1 or not title2: return False
    ratio = difflib.SequenceMatcher(None, title1, title2).ratio()
    return ratio > 0.6

def search_tavily(keyword, max_results=5):
    """Tavilyで検索（高品質）"""
    articles = []
    try:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        # ニュース検索モードを使用
        response = tavily.search(query=keyword, topic="news", max_results=max_results, include_images=True)
        
        for r in response.get('results', []):
            articles.append({
                "title": r.get('title'),
                "url": r.get('url'),
                "content": r.get('content'), # Tavilyは本文要約をきれいに返してくれる
                "image_url": r.get('image_url') or r.get('image'), # 画像も取得
                "search_source": keyword
            })
        print(f"✅ Tavily Search Success: {keyword}")
    except Exception as e:
        print(f"⚠️ Tavily Error ({keyword}): {e}")
    return articles

def search_ddg(keyword, max_results=5):
    """DuckDuckGoで検索（予備）"""
    articles = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(keyword, region="jp-jp", safesearch="off", max_results=max_results))
            for r in results:
                articles.append({
                    "title": r.get('title'),
                    "url": r.get('url'),
                    "content": r.get('body') or r.get('title'),
                    "image_url": r.get('image'),
                    "search_source": keyword
                })
        print(f"✅ DDG Search Success: {keyword}")
    except Exception as e:
        print(f"⚠️ DDG Error ({keyword}): {e}")
    return articles

def search_and_scrape(topic: str, user_topics: list, max_results: int = 6):
    search_keywords = []
    
    # トピック選択
    if topic in ["All", "For You"]:
        if user_topics:
            search_keywords = random.sample(user_topics, min(len(user_topics), 3))
        else:
            search_keywords = ["最新技術", "ビジネス", "トレンド"]
        print(f"🔍 Mixing Topics: {search_keywords}")
    else:
        search_keywords = [topic]
        print(f"🔍 Specific Topic: {search_keywords}")
    
    all_articles = []
    
    # 検索実行（Tavilyを優先し、だめならDDG）
    for keyword in search_keywords:
        results = []
        # まずTavilyを試す
        if TAVILY_API_KEY and "tvly-" in TAVILY_API_KEY:
             results = search_tavily(keyword, max_results=4)
        
        # Tavilyがだめ（またはキーがない）ならDDGを使う
        if not results:
            results = search_ddg(keyword, max_results=4)
            
        all_articles.extend(results)
        time.sleep(1)

    # 重複排除処理
    final_articles = []
    seen_urls = set()
    seen_titles = [] 

    random.shuffle(all_articles) 

    for article in all_articles:
        url = article['url']
        title = article['title']
        
        if url in seen_urls: continue
            
        is_duplicate = False
        for existing_title in seen_titles:
            if is_similar(title, existing_title):
                print(f"♻️ Skip similar: {title[:10]}... <=> {existing_title[:10]}...")
                is_duplicate = True
                break
        
        if is_duplicate: continue

        seen_urls.add(url)
        seen_titles.append(title)
        
        # 画像がない場合の補完
        if not article.get('image_url'):
            found_image = get_page_metadata(url)
            if found_image:
                article['image_url'] = found_image
        
        final_articles.append(article)
        if len(final_articles) >= max_results:
            break
            
    return final_articles