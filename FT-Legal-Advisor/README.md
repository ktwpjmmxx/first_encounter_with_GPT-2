# Guardian AI (Local Model Engine Version)

**Guardian AI** は、IT開発現場における法的リスク（偽装請負、下請法違反、著作権侵害など）を検知・診断するために特化した、ローカル動作型のAIアシスタントです。

本プロジェクトでは、**「論理的な法務思考（Logic）」** を獲得させることを目的に、Llama-3-8B モデルに対して特定の法務データセットを用いた Fine-Tuning（LoRA）を実施しました。

---

## Project Overview

従来の汎用LLMは、法律相談に対して「一般論」を返すか、あるいは「契約すればOK」といった誤った直感的回答をすることがありました。
本モデルは、**「たとえ合意があっても、強行法規（下請法など）に違反すればリスクである」** という専門家の判断プロセスを学習しています。

### Key Objectives
1.  **Alignment (倫理適合)**: ユーザーの利益になる提案であっても、違法性があれば明確に拒絶する。
2.  **Structured Output**: チャット形式ではなく、`リスクレベル` `関連法規` `分析` `修正案` の構造化データを出力する。
3.  **Local Execution**: 外部APIに依存せず、機密情報をローカル環境内で処理する。

---

## Model Architecture & Training

* **Base Model**: `Llama-3-ELYZA-JP-8b`
* **Method**: LoRA (Low-Rank Adaptation) via Unsloth
* **Dataset**: 
    * 3,000+ synthetic examples specific to IT law (SES, Subcontracting, Copyright, Privacy).
    * Generated via Gemini Pro, manually curated for "Risk Analysis" logic.
* **Infrastructure**: Google Colab (T4 GPU)

---

## Performance & Evaluation (Inference Logs)

実際の推論テスト（2026/01/30実施）において、本モデルは高度な論理判断能力を示しました。

### Success Case 1: 偽装請負の看破
**Scenario**: SES契約のエンジニアに対し、効率のためにチャットで直接指揮命令を行い、契約書で「指揮命令に従う」と合意させる。
> **AI Judgment**: **[Risk: High]**
> 「SES（準委任）は派遣ではないため、指揮命令権はない。契約書に記載しても、実態が指揮命令であれば偽装請負（労働者派遣法違反）となる」と正しく指摘。契約の形式よりも実態を優先する法的思考ができている。

### Success Case 2: 下請法の強行法規性
**Scenario**: 業績悪化のため、下請業者の合意を得て代金を10%減額する。
> **AI Judgment**: **[Risk: High]**
> 「合意があっても、発注者都合の減額は下請法（減額の禁止）違反となる」と判定。素人が陥りやすい「合意＝合法」の罠を回避している。

### Limitation Case: 著作権法の条文知識 (Need for RAG)
**Scenario**: AI学習目的の画像スクレイピングの適法性判断。
> **AI Judgment**: **[Risk: Medium]**
> 営利目的のリスクを指摘するなど結論は妥当だが、根拠として引用した条文が「第30条（私的利用）」など、AI学習に特化した最新条文（第30条の4）とズレているケースが確認された。

---

## 技術的限界点と今後の展望について

今回の検証により、以下の技術的限界が特定されました。

### The "Parametric Memory" Problem
8Bパラメータのモデルにとって、膨大な条文番号や最新の法改正情報を「正確に暗記（Parametric Memory）」し続けることは困難です。
論理（Logic）は正しいものの、条文番号のような事実（Fact）においてハルシネーション（幻覚）が発生するリスクが残ります。

### Solution: RAG Architecture
次期フェーズ（Phase 3）では、**「思考と記憶の分離」** を行います。

* **Logic (思考)**: 本モデル (Fine-Tuned Llama-3) が担当。
* **Memory (記憶)**: 外部の法令データベース (Vector DB) が担当。

RAG（Retrieval-Augmented Generation）を導入し、正確な条文を検索してモデルに提示することで、論理の正しさと情報の正確性を両立させます。

---

##　Installation & Usage

本リポジトリは、学習済みモデルをローカルで動作させるための推論アプリを含んでいます。

### Requirements
* Python 3.10+
* GPU (Recommended: VRAM 12GB+)
* Dependencies: `unsloth`, `streamlit`, `torch`

### Quick Start
```bash
# 1. Install dependencies
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install streamlit

# 2. Run the application
streamlit run app_local.py