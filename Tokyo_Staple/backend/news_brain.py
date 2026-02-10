import math
import re
import os
import random
import json
from typing import List, Dict, Optional
from groq import Groq

# ▼▼▼ あなたのAPIキーを入力してください ▼▼▼
GROQ_API_KEY = "YOUR_API_KEY" 
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

def calculate_read_time(text):
    if not text: return "1 min read"
    minutes = math.ceil(len(text) / 500)
    return f"{minutes} min read"

def analyze_news(articles: List[Dict], topic: str = "All", preferences: Optional[Dict] = None) -> List[Dict]:
    if not articles: return []
    
    user_context = ""
    if preferences:
        liked = ", ".join(preferences.get("liked_tags", []))
        if liked:
            user_context += f"【読者層】{liked} に関する深い専門知識やビジネス的な洞察を好みます。"

    client = None
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"❌ Groq初期化エラー: {e}")

    analyzed_articles = []
    print(f"🚀 AI執筆モード起動 (Topic: {topic})...")

    for article in articles:
        original_text = article.get('content', '')[:6000]
        
        # 画像がない場合のプレースホルダー
        img_url = article.get("image_url")
        if not img_url:
            placeholders = [
                "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&q=80&w=1000", 
                "https://images.unsplash.com/photo-1495020689067-958852a7765e?auto=format&fit=crop&q=80&w=1000",
                "https://images.unsplash.com/photo-1526304640152-d4619684e484?auto=format&fit=crop&q=80&w=1000",
            ]
            img_url = random.choice(placeholders)

        generated_title = article['title']
        generated_summary = ""
        generated_content = ""
        
        try:
            if not client: raise Exception("Client not initialized")

            # ★修正されたプロンプト：テンプレ禁止・多様性重視
            system_prompt = f"""
            あなたは高級経済誌（Wired, Newspicks, HBRなど）の敏腕編集長です。
            断片的なニュース情報を元に、読者を惹きつける「完成された記事」を作成してください。

            【タイトル作成ルール】
            1. 日本語で**35文字以内**。スマホ2行以内。
            2. 言い切り型で、インパクト重視。「...」で終わらせない。

            【本文作成ルール (重要)】
            1. **ボリューム**: 1000文字以上の長文で書くこと。
            2. **見出しの多様性 (最重要)**: 
               - 「背景」「結論」「核心」といった**定型句は使用禁止**。
               - 記事の内容に合わせて、独自の魅力的な小見出しをつけること。
               - 例: 「市場への激震」「隠されたリスク」「技術的なブレイクスルー」「競合他社の動き」など。
            3. **構造**: 以下のHTMLタグのみ使用。
               - `<h3>`: 小見出し
               - `<p>`: 本文段落
            
            出力は必ずJSON形式のみ。
            """

            user_prompt = f"""
            以下の情報を元に記事を作成してください。

            元タイトル: {article['title']}
            元内容: {original_text}

            {{
                "new_title": "35文字以内のタイトル",
                "editorial": "100文字程度の要約",
                "full_story": "<h3>（ユニークな見出し）</h3><p>...</p>..."
            }}
            """

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
            )

            raw_response = chat_completion.choices[0].message.content
            result_json = json.loads(raw_response)
            
            generated_title = result_json.get("new_title", article['title'])
            generated_summary = result_json.get("editorial", "")
            generated_content = result_json.get("full_story", "")

        except Exception as e:
            print(f"❌ AI生成エラー: {e}")
            generated_summary = original_text[:100] + "..."
            generated_content = f"<p>（※エラー: {e}）</p><p>{original_text}</p>"

        read_time = calculate_read_time(generated_content)

        analyzed_articles.append({
            "title": generated_title,
            "url": article['url'],
            "image_url": img_url,
            "editorial": generated_summary,
            "full_story": generated_content,
            "read_time": read_time,
            "source_badge": "AI EDITORIAL",
            "search_source": article.get("search_source", "TOPIC")
        })

    return analyzed_articles