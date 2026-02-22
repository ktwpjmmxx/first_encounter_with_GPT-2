import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai

# ガードレール機能の読み込み
from input_filter import InputFilter

# 環境変数（.env）の読み込み
load_dotenv()

# Gemini APIの設定
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ 警告: GOOGLE_API_KEY が .env ファイルに設定されていません。")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

# CORS設定（Reactフロントエンドからの通信を許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発用。本番環境ではフロントエンドのURLを指定します
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 入力フィルタの初期化
input_filter = InputFilter()

# リクエストのデータ構造定義（App.jsxの { text: inputText } に対応）
class AssessmentRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Guardian AI Backend is Running"}

@app.post("/api/assess")
def assess_compliance(request: AssessmentRequest):
    print(f"診断リクエスト受信: {request.text[:30]}...")

    # 1. ガードレールによるスコープチェック
    is_in_scope, message, category = input_filter.check_scope(request.text)
    
    if not is_in_scope:
        print(f"スコープ外を検知 ({category}): {message}")
        # スコープ外の場合は、APIコストをかけずに即座にダミー結果を返す
        return {
            "risk_score": 0,
            "risk_level": "Low",
            "summary": "診断対象外の入力です",
            "laws": [],
            "reason": message,
            "recommendations": []
        }

    # 2. Geminiによる法的リスク診断
    try:
        # Gemini 2.5 Flash モデルを使用 (最新の軽量・高速モデル)
        # ※もしエラーが出る場合は "gemini-1.5-flash" に変更してください
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        system_prompt = """
        あなたはGuardian AI、ソフトウェア開発の仕様書から法的リスクを分析する専門的な法務AIアシスタントです。
        ユーザーの入力内容から、日本の法律（個人情報保護法、資金決済法、特定商取引法、著作権法など）やGDPRなどの観点でリスクを評価してください。
        
        必ず以下のJSONスキーマに従って、有効なJSON形式で出力してください。Markdownのコードブロック(```json ... ```)は付けずに、純粋なJSONテキストのみを返してください。
        
        {
          "risk_score": 0〜100の整数 (100が最も危険),
          "risk_level": "High" または "Medium" または "Low",
          "summary": "リスクの全体的な概要（1文程度）",
          "reason": "AIによる詳細な分析理由（2〜3文）",
          "laws": [
            {"label": "関連する法律名", "category": "カテゴリ名（例: Privacy, Finance, Copyright）"}
          ],
          "recommendations": [
            {
              "title": "対策のタイトル（短く）",
              "priority": "High" または "Medium" または "Low",
              "description": "具体的な対策の解説"
            }
          ]
        }
        """

        print("Gemini AIに分析を依頼中...")
        response = model.generate_content(
            contents=[
                {"role": "user", "parts": [system_prompt + "\n\n【分析対象の仕様】\n" + request.text]}
            ],
            generation_config=genai.types.GenerationConfig(
                # JSONモードを強制して、パースエラーを防ぐ
                response_mime_type="application/json",
                temperature=0.2 # 堅い回答を求めるため低めに設定
            )
        )

        # JSONテキストをPythonの辞書に変換
        result_data = json.loads(response.text)
        print("✅ 診断完了！")
        
        return result_data

    except Exception as e:
        print(f"❌ Gemini API エラー: {e}")
        # エラーが発生した場合も、フロントエンドがクラッシュしないように安全な形式で返す
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)