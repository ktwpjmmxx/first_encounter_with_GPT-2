import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv

# 既存のフィルタリングモジュールを読み込み
# (input_filter.py が同じディレクトリにある前提)
from input_filter import InputFilter

load_dotenv()

app = FastAPI()

# ★ CORS設定 (Reactからのアクセスを許可)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React(Vite)のデフォルトポート
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini設定
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')
input_filter = InputFilter()

# リクエストボディの定義
class AssessmentRequest(BaseModel):
    text: str

@app.post("/api/assess")
async def assess_risk(request: AssessmentRequest):
    user_input = request.text

    # 1. フィルタリングチェック
    is_valid, msg, category = input_filter.check_scope(user_input)
    if not is_valid:
        # 範囲外の場合はエラーではなく、判定不能として返す
        return {
            "risk_score": 0,
            "risk_level": "Out of Scope",
            "summary": msg,
            "laws": [{"label": category, "category": "Scope"}],
            "reason": msg,
            "recommendations": []
        }

    # 2. Geminiへのプロンプト (Figrのデザインに合わせてスコアを追加)
    prompt = f"""
    あなたは法務リスク診断AI「Guardian AI」です。
    以下の仕様の法的リスクを診断し、必ずJSON形式で出力してください。

    【仕様】
    {user_input}

    【出力フォーマット(JSON)】
    {{
        "risk_score": 0〜100の整数 (高いほど高リスク),
        "risk_level": "High/Medium/Low",
        "summary": "20文字以内の概要",
        "laws": [{{"label": "関連法名(例: GDPR)", "category": "カテゴリ(例: Privacy)"}}],
        "reason": "専門的な詳細分析",
        "recommendations": [
            {{"title": "アクション概要", "priority": "high/medium/low", "description": "詳細な説明"}}
        ]
    }}
    """

    try:
        response = model.generate_content(prompt)
        # JSON部分のみ抽出してパース
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(response_text)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)