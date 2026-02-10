from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import json

# 自作のAIモジュール
from search_agent import search_and_scrape
from news_brain import analyze_news

app = FastAPI()

# CORS設定（Reactからの通信を許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 評価データの保存先ファイル
PREFERENCES_FILE = "user_preferences.json"

# リクエストの型定義
class NewsRequest(BaseModel):
    topic: str
    user_topics: List[str]

class RatingRequest(BaseModel):
    article_title: str
    rating: int
    tags: List[str]

# 起動時に設定ファイルを読み込む
def load_preferences():
    if not os.path.exists(PREFERENCES_FILE):
        return {"liked_tags": [], "disliked_tags": []}
    try:
        with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"liked_tags": [], "disliked_tags": []}

def save_preferences(data):
    with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "News AI Backend is Running"}

@app.post("/api/news")
def generate_news(request: NewsRequest):
    print(f"📩 注文受信: トピック '{request.topic}'")
    
    # ユーザーの好みを読み込む
    prefs = load_preferences()
    print(f"👤 現在のユーザーの好み: {prefs['liked_tags'][:3]}...")

    # 1. 記事を探す
    raw_articles = search_and_scrape(request.topic, request.user_topics)
    
    # 2. AIによる加筆修正
    analyzed_articles = analyze_news(raw_articles, request.topic, prefs) 
    
    print(f"✅ 生成完了: {len(analyzed_articles)}件の記事を返却します")
    return {"articles": analyzed_articles}

# ★新機能: 評価を受け取るAPI
@app.post("/api/feedback")
def submit_feedback(request: RatingRequest):
    print(f"⭐ 評価受信: {request.article_title} -> {request.rating}点")
    
    prefs = load_preferences()
    
    # ロジック: ★4以上は「好き」、★2以下は「嫌い」にカウント
    if request.rating >= 4:
        for tag in request.tags:
            if tag not in prefs["liked_tags"]:
                prefs["liked_tags"].append(tag)
                print(f"👍 '{tag}' を好きリストに追加しました")
                
    elif request.rating <= 2:
        for tag in request.tags:
            if tag not in prefs["disliked_tags"]:
                prefs["disliked_tags"].append(tag)
                print(f"👎 '{tag}' を嫌いリストに追加しました")
    
    save_preferences(prefs)
    return {"status": "success", "current_prefs": prefs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)