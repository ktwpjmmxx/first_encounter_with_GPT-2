# 🤖 GPT-2チャットボット開発のための思考案 - Google Colab版

## 📋 思考案の構成

- **①～Ⓧ**のテーマ
- **●コードブロック全体**（コード全体をテーマごとに区切った塊）
  - 【コードの小テーマ】
  - 該当部分の抜粋コード
  - 抜粋コードについての説明
- **◆コードブロック全体の解釈**
- **備忘録**

---

## ① Google Colabのための環境設定

### 🔧 コードブロック1

```bash
!pip install transformers==4.21.0 torch==1.12.1 datasets==2.4.0 accelerate==0.12.0
!pip install sentencepiece==0.1.97 sacremoses==0.0.53
print("ライブラリのインストールが完了しました。")
```

### 📦 コード概要

#### `!pip install`

**transformers==4.21.0**
→ Hugging FaceのTransformersライブラリ。
GPT-2などの事前学習済みモデルやトークナイザーを使うのに必須。
※バージョン4.21.0で環境を固定しているのは、コードの互換性を保つため。

**torch==1.12.1**
→ PyTorchのバージョン。深層学習モデルの計算ライブラリ。
　GPUを使った高速処理を担う。

**datasets==2.4.0**
→ Hugging FaceのDatasetsライブラリ。データセットの読み込みや加工を簡単に行える。

**accelerate==0.12.0**
→ モデルの分散学習や高速化を簡単に実装するためのユーティリティ。
　ColabのGPUを有効活用する際に便利。

---

**sentencepiece==0.1.97**
→ Googleが開発したサブワード分割ツール。
　日本語などの言語でトークナイズ（単語分割）するときに使用。

**sacremoses==0.0.53**
→ MosesトークナイザーのPython版。
　文章の前処理（トークン分割や正規化）で使われる。
　英語のトークナイザーとして多用されているが、日本語モデルでも使うケースあり。

### ◆ コードブロック1の全体の解釈

- **1行目** ... Transformerモデルや学習処理の基盤ライブラリをインストール
- **2行目** ... 日本語テキストを分割・処理するためのツールを追加でインストール

#### 📝 【備忘録】

**BPEのはずがなぜトークナイザーに「sentencepiece」や「sacremoses」があるのか**

- GPT-2の標準トークナイザーは `GPT2Tokenizer` （BPEベース）で完結
- `sentencepiece` は主に日本語や多言語対応のモデル（T5やmBERTなど）が使う別のトークナイザー
- `sacremoses` は主に英語などの形態素解析やトークナイズ前のテキスト正規化で使われる

---

### 🛠️ コードブロック2

```python
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import re
import random
from typing import List, Dict, Optional
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU使用: {torch.cuda.get_device_name(0)}")
    print(f"利用可能VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    device = torch.device('cpu')
    print("CPU使用（警告: 学習に時間がかかります）")

torch.manual_seed(42)
random.seed(42)
```

### 📚 コード概要

#### 【インポート部分】

```python
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import re
import random
from typing import List, Dict, Optional
import warnings
from datetime import datetime
import os
```

**GPT2LMHeadModel**
→ GPT-2の言語モデル本体。

**GPT2Tokenizer**
→ GPT-2用のトークナイザー。

**TrainingArguments, Trainer**
→ モデルの学習設定と学習ループ管理を簡単にするクラス。

**DataCollatorForLanguageModeling**
→ 言語モデル用のデータ整形ユーティリティ。

**datasets.Dataset**
→ Hugging Faceのデータセットラッパー。

**json, re, random, typing, warnings, datetime, os**
→ それぞれデータ処理、型ヒント、警告管理、日時操作、ファイル操作などで使う一般ライブラリ。

#### 【警告非表示】

```python
warnings.filterwarnings('ignore')
```
→ 開発時の余計な警告を隠す

#### 【デバイスの判定と設定】

```python
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU使用: {torch.cuda.get_device_name(0)}")
    print(f"利用可能VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    device = torch.device('cpu')
    print("CPU使用（警告: 学習に時間がかかります）")
```

- GPUが使えるかどうかチェックして、使えるならGPU（cuda）、なければCPUを選択。
- GPUなら名前とVRAM容量を表示してくれるので、環境の確認に便利。
- CPUのみの場合は「時間かかります」と警告。

#### 【乱数シード固定】

```python
torch.manual_seed(42)
random.seed(42)
```

**① torch.manual_seed(X)**
→ PyTorchの乱数生成器。モデルの重み初期化、テンソル生成(torch.rand/torch.randnなど)

**② random.seed(42)**
→ Pythonの標準乱数生成器。random.randint, random.shuffleなどPythonの標準乱数

#### ◆ 2つ必要な意味

**データ関連** → Pythonのrandomを使う（例：学習データをシャッフル）

**モデルやテンソル関連** → PyTorchの乱数を使う（例：重み初期化）

両方で乱数が使われるから、両方のシードを固定しないと完全再現できない。

- `torch.manual_seed(x)` → モデル内部のランダム性を固定
- `random.seed(x)` → Python側のランダム性を固定

### ◆ コードブロック2の全体の解釈

土台つくりの部分。GPU/CPU判定で計算環境を設定し、結果のブレを防ぐために乱数シードを固定する。

#### 📝 【備忘録】

**乱数シードを固定する意味**

深層学習では、重みの初期化やデータシャッフルなどに乱数が使われる

もし、シード値を固定しないと...
1. **毎回モデルの初期状態が違う** → 学習結果が毎回微妙に変わる
2. **結果の再現性がなくなる** → 他人や未来の自分が同じ結果を再現できない

といった問題が起こる

シード値は「乱数列を作る際のスタート番号」みたいなもの
同じシード値を使えば、同じ乱数列が生成されるので結果も同じになる

#### 🌌 【42の由来】

元ネタは、ダグラス・アダムスのSF小説
『銀河ヒッチハイク・ガイド』（The Hitchhiker's Guide to the Galaxy）

物語の中で、超スーパーコンピュータ「ディープ・ソート」に
「生命、宇宙、そして万物の究極の疑問の答え」を計算させたら、
答えが **"42"** だった というシュールなオチ。

その意味は最後まで明かされず、「答えは42」というジョークとして有名に。

---

## ② モデルとトークナイザーの詳細設定

### 🎯 GPT-2のサイズ

1. **GPT-2 (124M パラメータ)** : 最小サイズ、Colabで安定動作
2. **GPT-2-medium (355M パラメータ)** : より高品質だがメモリ使用量大
3. **GPT-2-large (774M パラメータ)** : さらに高品質だがColab無料版では厳しい

### 🤖 コードブロック3

```python
model_name = "gpt2"
print(f"モデル '{model_name}' を読み込み中...")

try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("パディングトークンをEOSトークンに設定しました。")
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"モデル読み込み完了!")
    print(f"総パラメータ数: {total_params:,}")
    print(f"学習対象パラメータ数: {trainable_params:,}")

except Exception as e:
    print(f"モデル読み込みエラー: {e}")
    print("インターネット接続またはHugging Faceのサーバー状況を確認してください。")
```

### 📊 コード概要

#### 【モデルサイズの選択】

```python
model_name = "gpt2"
print(f"モデル '{model_name}' を読み込み中...")
```

→ `model_name` に "gpt2" を指定
Hugging Faceのモデル名で、この場合は一番小さいGPT-2（約1.24億パラメータ）を使う
他には"gpt2-medium", "gpt2-large", "gpt2-xl" などがある

#### 【トークナイザーの読み込み】

```python
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
```
→ Hugging FaceからGPT-2用のBPEトークナイザーをロード
　`from_pretrained`...ネットからモデルや辞書ファイルをダウンロードしてキャッシュに保存

#### 【パディングトークンの設定】

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("パディングトークンをEOSトークンに設定しました。")
```

→ GPT-2にはパディング用トークンが存在しない（BERTみたいな[PAD]なし）
　バッチ学習で長さを揃えるため、代わりに文末トークン（eos_token）を流用する

#### 【モデル本体の読み込み】

```python
model = GPT2LMHeadModel.from_pretrained(model_name)
```
→ GPT-2の「言語モデル」部分をロード（LMHeadは単語予測用の最終層を含む構造）
　事前学習済みの重みも一緒に読み込む

#### 【デバイスに移動】

```python
model.to(device)
```
→ GPUがあればGPUメモリへ転送して計算を高速化。CPUの場合はメモリ上で計算（遅くなる）。

#### 【パラメータ数の表示】

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

→ **総パラメータ数**: モデル全体の重みの数。

　**学習対象パラメータ数**: 勾配計算が有効（requires_grad=True）なパラメータ数。

　転移学習や微調整（fine-tuning）の時、固定されてる層があるとこの値は減る。

#### 【例外処理】

```python
except Exception as e:
    print(f"モデル読み込みエラー: {e}")
    print("インターネット接続またはHugging Faceのサーバー状況を確認してください。")
```

→ モデルやトークナイザーのロードが失敗したときにエラー原因を表示。
　ネット接続やHugging Face側の障害もここで検知。

### ◆ コードブロック3の全体の解釈

全体的に初期のセットアップ工程

1. **モデルサイズ選択**
2. **トークナイザー準備**（パディング設定含む）
3. **モデル読み込み**
4. **デバイス転送 & パラメータ数確認**

#### 📝 【備忘録】

**パディングについて**

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**GPT-2はpad_tokenを持ってない**
- GPT-2は言語生成モデルなので、「途中で埋める」という発想がなく、`pad_token` が None（未定義）。なので、このまま`padding=True`で使おうとするとエラーになる

**EOSトークンを代わりに使う理由**
- GPT-2では**`<eos>`（End Of Sequence）**が「文の終わり」を表す唯一の特別トークン。
- バッチ入力でパディングが必要な場合、`pad_token` の代わりに `eos_token` を使うのが一般的な対処法

- 言語モデルは基本的に「終わった後の空白部分」は学習に影響しにくい（attention_maskで無視できる）
- pad専用トークンを作るより、既存のEOSを使ったほうがシンプルで安全

**まとめ**
- GPT-2には`pad_token`がない → 代わりに`eos_token`を設定
- これで`padding=True`が使えるようになり、Trainerやデータローダーがバッチ化できる
- 実際の計算では`attention_mask`がpad部分を無視する

**パラメーター数の表示について**

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```