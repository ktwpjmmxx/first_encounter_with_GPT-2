# Guardian AI - Legal Compliance Intelligence

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Frontend](https://img.shields.io/badge/Frontend-React_Vite-61DAFB)
![Backend](https://img.shields.io/badge/Backend-FastAPI_Python-009688)
![AI](https://img.shields.io/badge/AI-Gemini_2.5_Flash-orange)

**開発仕様書から法的リスクを瞬時に可視化する、SaaS型 法務チェックプラットフォーム**

## Overview

**Guardian AI** は、ソフトウェア開発の初期段階（設計フェーズ）において、法的リスクを早期発見するためのAIアプリケーションです。

「仕様書」や「機能概要」を入力するだけで、Google Gemini 2.5 Flash が関連する法律（GDPR、資金決済法、個人情報保護法など）を照合し、リスクスコアと具体的な改善案を提示します。

本プロジェクトは、モダンなWebアプリケーション開発（React/FastAPI）と、生成AIの実践的な統合（Prompt Engineering/JSON Mode）を実証するために開発されました。

## Key Features

* **Risk Scoring Dashboard**
  法的リスクを「0〜100」のスコアで定量化。ゲージチャートで直感的に安全性を把握できます。

* **Real-time Analysis**
  FastAPI と Gemini Flash モデルの連携により、数秒で診断結果が返ってきます。

* **Scope Filtering**
  AIが苦手とする領域（OSSライセンスや倫理問題）をフィルタリングし、ハルシネーションを防ぐガードレール機能を実装。

* **Modern UI/UX**
  React + Tailwind CSS による、ダークモード基調の洗練されたダッシュボードデザイン。

## Tech Stack

| Category | Technology | Usage |
| :--- | :--- | :--- |
| **Frontend** | React, Vite | 高速なSPA（Single Page Application）構築 |
| **Styling** | Tailwind CSS, Lucide | モダンなコンポーネント設計とアイコン |
| **Backend** | FastAPI (Python) | 非同期処理による高速なAPIサーバー |
| **AI Model** | Google Gemini 2.5 Flash | 高速・低コストな推論、JSONモードの使用 |
| **Validation** | Pydantic | 堅牢なデータバリデーション |

## Architecture

フロントエンドとバックエンドを疎結合にし、スケーラビリティを意識した構成です。

```mermaid
graph LR
    User[User / Browser] -->|Input Specs| React[React Frontend]
    React -->|JSON Request| API[FastAPI Backend]

    subgraph "Backend Logic"
        API -->|1. Scope Check| Filter[Input Filter]
        Filter -->|2. Validated Text| LLM[Gemini 2.5 Flash]
        LLM -->|3. Risk Analysis (JSON)| API
    end

    API -->|Response (Score & Advice)| React
```

## Directory Structure

```text
Guardian-AI/
├── backend/                 # API Server
│   ├── main.py              # Entry point
│   ├── input_filter.py      # Guardrails logic
│   └── requirements.txt     # Python dependencies
├── frontend/                # Client App
│   ├── src/
│   │   ├── App.jsx          # Main Dashboard UI
│   │   └── ...
│   └── package.json         # JS dependencies
└── README.md
```

## Quick Start

本アプリケーションは、バックエンドとフロントエンドの両方を起動する必要があります。

### Prerequisites

* Python 3.10+
* Node.js 18+
* Google API Key

### 1. Backend Setup (FastAPI)

```bash
cd backend

# 仮想環境の作成と有効化 (任意)
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt

# 環境変数の設定
# .envファイルを作成し、GOOGLE_API_KEY=your_key を記述してください

# サーバー起動 (http://localhost:8000)
python main.py
```

### 2. Frontend Setup (React)

新しいターミナルを開いて実行してください。

```bash
cd frontend

# パッケージインストール
npm install

# 開発サーバー起動 (http://localhost:5173)
npm run dev
```

ブラウザで `http://localhost:5173` にアクセスすると、Guardian AI が利用可能です。

## Future Roadmap

* [ ] **Custom Knowledge Base (RAG)**: 独自の社内規定やガイドラインを読み込ませる機能。
* [ ] **PDF Report Export**: 診断結果を法務部門提出用のPDFとして出力。
* [ ] **Authentication**: Auth0 または Supabase を利用したユーザー管理。

## Author

* **Role:** Full Stack AI Engineer
* **Focus:** Legal Tech, Generative AI Application Development
