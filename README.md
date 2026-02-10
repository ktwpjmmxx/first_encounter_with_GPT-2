# Applied GenAI Projects

生成AI技術（LLM）を活用したアプリケーション開発のポートフォリオです。

LLMのファインチューニング（Fine-Tuning）から、APIを活用したRAG（Retrieval-Augmented Generation）構築、Webアプリケーションとしての実装まで、一気通貫した開発実験を行っています。

## Projects Overview

| Project Name | Description | Key Tech Stack |
| :--- | :--- | :--- |
| **[GuardianAI-FT](./FT-Legal-Advisor)** | **IT法務特化 契約書チェックAI**<br>独自データセットを用いてLLMをファインチューニングし、IT法務に特化した契約書のレビュー・修正提案を行うモデルを作成・検証しました。 | <img src="https://img.shields.io/badge/-Fine--Tuning-red" /> Python, Llama 3 (or User's Model), PyTorch |
| **[GuardianAI-API](./API-Legal-Advisor)** | **法務チェック・アシスタント (API版)**<br>OpenAI API等を活用し、プロンプトエンジニアリングとRAGによって法務文書のチェックを行うツール。FT版との精度比較やコスト検証も行っています。 | <img src="https://img.shields.io/badge/-OpenAI%20API-green" /> <img src="https://img.shields.io/badge/-LangChain-blue" /> Python, Streamlit |
| **[TokyoStaple](./TokyoStaple)** | **パーソナライズニュースアプリ**<br>ユーザーの好みを学習し、最適なニュースを配信するWebアプリケーション。生成AIを用いて記事の要約やタグ付けを自動化しています。 | <img src="https://img.shields.io/badge/-Next.js-black" /> <img src="https://img.shields.io/badge/-Python-yellow" /> Supabase, Vercel |

---

## Technical Highlights

このポートフォリオで扱っている主な技術スタックです。

* **Generative AI:**
    * **Fine-Tuning:** カスタムデータセット作成, LoRA/QLoRAによる学習 (GuardianAI)
    * **RAG / Prompt Engineering:** LangChain, Vector DB (Chroma/Pinecone), OpenAI API
* **Backend / ML:**
    * Python (FastAPI / Flask), PyTorch, Hugging Face Transformers
* **Frontend / Web:**
    * Next.js, TypeScript, Streamlit (プロトタイピング)
* **Infrastructure / Cloud:**
    * Docker, GCP (Compute Engine), Vercel, Supabase

## Detailed Documentation

各プロジェクトの詳細なセットアップ方法、技術選定の背景、デモ動画などは、各ディレクトリ内の `README.md` をご参照ください。

* **[GuardianAI-Legal-FT の詳細へ](./FT-Legal-Advisor)**
* **[Legal-Check-API の詳細へ](./API-Legal-Advisor)**
* **[TokyoStaple の詳細へ](./TokyoStaple)**

---

## Author

* **Name:** [Tatsuya Koyama / ktwpjmmxx]
* **Focus:** AI Engineer / Legal Tech Developer
* **Contact:** [mmxxv15t@gmail.com]