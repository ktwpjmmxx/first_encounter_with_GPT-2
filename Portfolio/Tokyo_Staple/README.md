# TOKYO STAPLE - AI-Powered Personalized News App

"自分の専属記者" が、世界中のニュースを再編集して届けるAIニュースアプリ。

<img src="screenshots/demo_top.jpg" width="100%" alt="Top Page Demo">

## Overview
TOKYO STAPLE は、ユーザーの好みを学習し、単なるリンク集ではなく「読み応えのある雑誌記事」としてニュースを再生成するアプリケーションです。

通常のニュースアプリと異なり、AI（Llama 3.3）が収集した断片的な情報を元に、「背景・詳細・影響」を含む1000文字以上の長文記事をリアルタイムで執筆します。また、ユーザーの評価を学習し、使えば使うほど自分好みのトピックや切り口に進化していく「成長するニュースフィード」を実現しました。

## Key Features

### 1. AI Editorial Engine
* LLM (Llama 3.3 70B) を活用し、断片的なウェブニュースを「洞察に富んだ長文記事」にリライト。
* 既存のニュース要約にありがちな「短すぎる・無機質」な問題を解決するため、プロンプトエンジニアリングにより「雑誌のような文体と構成（HTML構造）」を強制。
* 記事タイトルも、スマホで読みやすい35文字以内のキャッチーな見出しに自動修正。

### 2. Smart Personalization
* ユーザーが記事に対して行った「5段階評価」をバックエンドで蓄積。
* 次回のニュース生成時に、高評価のタグをAIのコンテキスト（短期記憶）に注入することで、検索キーワードと解説の切り口を動的に変化させます。
* 例: 「SpaceX」の記事に高評価 → 次回から宇宙関連のニュースが増え、技術的な解説が深くなる。

### 3. Intelligent Search Aggregator
* Tavily API (AI特化検索) と DuckDuckGo を併用したハイブリッド検索システム。
* Semantic De-duplication: difflib を用いた類似度判定により、異なるメディアが報じた同じ内容のニュース（例: "SpaceX買収"）が重複して表示されるのを防ぎ、フィードの多様性を確保。

### 4. High-Fidelity UI
* 「読む体験」を最優先し、紙媒体のような明朝体（Shippori Mincho）と可読性の高いゴシック体を使い分け。
* React + Tailwind CSS (Typography) により、美しくフォーマットされた記事レイアウトを実現。

---

## Tech Stack

| Category | Technology |
| --- | --- |
| **Frontend** | React (Vite), Tailwind CSS, Lucide React |
| **Backend** | Python, FastAPI, Uvicorn |
| **AI Model** | Groq (Llama-3.3-70b-versatile) |
| **Search** | Tavily API, DuckDuckGo Search (Fallback) |
| **Data** | JSON (User Preferences Storage) |

---

## Directory Structure

本リポジトリは、主要なコードのみを抽出した構成となっています。

```text
Tokyo_Staple/
│
├── backend/                   # Python側のコード
│   ├── main.py                # サーバー本体
│   ├── news_brain.py          # AIロジック
│   ├── search_agent.py        # 検索ロジック
│   └── requirements.txt       # 必要なライブラリ一覧
│
├── frontend/                  # React側のコード
│   ├── src/
│   │   ├── App.jsx            # メイン画面とロジック
│   │   ├── main.jsx           # エントリーポイント
│   │   └── index.css          # TailwindなどのCSS設定
│   ├── index.html             # フォント読み込みタグがあるファイル
│   ├── tailwind.config.js     # デザイン設定
│   ├── postcss.config.js      # PostCSS設定
│   ├── vite.config.js         # ビルド設定
│   └── package.json           # ライブラリ情報の定義書
│
├── screenshots/               # デモ画像置き場
│   ├── demo_top.jpg           # トップページの画像
│   ├── demo_article_1.jpg     # 記事詳細の画像(1/2)
│   ├── demo_article_1-2.jpg   # 記事詳細の画像(2/2)
│   ├── demo_stars.jpg         # 記事の星5段階の評価画面
│   └── demo_setting.jpg       # トピック編集画面
│
└── README.md                  # プロジェクトの説明書
```

---

## Technical Highlights

### 1. 検索と生成の分離設計
「検索（Fact）」と「生成（Opinion）」のプロセスを明確に分離しました。
search_agent.py が事実情報を収集し、news_brain.py がそれを解釈・執筆するというパイプラインを構築することで、ハルシネーション（嘘の生成）を抑制しつつ、読み物としての面白さを追求しています。

### 2. 動的プロンプトインジェクション
ユーザーのフィードバック（星5段階による記事の評価）をLLMへのシステムプロンプトに直接埋め込む設計にしました。
これにより、「単にその単語を含む記事を探す」だけでなく、「ユーザーが好む視点（例：ビジネス戦略寄り、技術詳細寄り）」で記事を執筆させることを可能にしました。

### 3. フォールバック戦略
外部API（Tavily）や画像リンク切れなどの「外部要因によるエラー」を前提とした設計を行いました。
* Tavilyがダウンした場合は即座にDuckDuckGoに切り替え。
* 画像が取得できない、またはブロックされた場合は、自動的にUnsplashの高品質なプレースホルダー画像に差し替え。
* これにより、デモ環境での安定性を高めています。

---

## How to Run

### Backend (Python)
```bash
cd backend
pip install -r requirements.txt
python main.py
```
APIサーバーが http://127.0.0.1:8000 で起動します。

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
ブラウザで http://localhost:5173 にアクセスしてください。

---

## 今後の展望

商用利用を鑑みた場合、以下の機能拡張と最適化が考えられます。

### 1. コストとパフォーマンスの最適化
LLMおよび検索APIの従量課金コストを削減し、レスポンス速度を向上させるためにキャッシュ層を導入します。
* **Redis Implementation**: 同一トピックや類似の検索クエリに対しては、生成済みの記事をRedis等のインメモリDBにキャッシュし、APIリクエスト回数を抑制します。
* **TTL Strategy**: ニュースの鮮度を保つため、キャッシュの有効期限（TTL）を適切に設定し、「コスト削減」と「情報の即時性」のバランスを最適化します。

### 2. サービス品質の向上
現在はプロトタイプとして無料枠や軽量モデルを使用していますが、実運用環境では有償サービスの導入により品質を担保します。
* **Advanced Models**: GPT-4o や Claude 3.5 Sonnet などの商用APIを採用し、より複雑な文脈理解と自然な日本語生成、ハルシネーションの低減を実現します。
* **Premium Search**: Google Custom Search API (Paid) などを導入し、検索ソースの信頼性と網羅性を向上させ、マイナーなトピックへの対応力を強化します。

---

## Note
このプロジェクトはデモアプリケーションです。APIキー等は環境変数またはコード内で適切に設定する必要があります。