!pip install transformers==4.21.0 torch==1.12.1 datasets==2.4.0 accelerate==0.12.0
!pip install sentencepiece==0.1.97 sacremoses==0.0.53
print("ライブラリのインストールが完了しました。")

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

def create_comprehensive_dataset():
"""
多様なシナリオをカバーする会話データセットを作成
各カテゴリーから自然な会話パターンを収集
"""
conversations = [

{"user": "おはようございます", "bot": "おはようございます！今日も良い一日をお過ごしください。何かお手伝い
{"user": "こんにちは", "bot": "こんにちは！お疲れさまです。今日はどのようなことについてお話しましょうか？
{"user": "こんばんは", "bot": "こんばんは！一日お疲れさまでした。リラックスしてお話しましょう。"},

{"user": "元気ですか？", "bot": "はい、とても元気です！あなたはいかがですか？今日は調子はどうでしょう？"
{"user": "疲れました", "bot": "お疲れさまです。少し休憩を取られてはいかがでしょうか？何か気分転換になるこ
{"user": "嬉しいことがありました", "bot": "それは素晴らしいですね！よろしければ、どのような嬉しいことがあ

{"user": "今日は何をしていましたか？", "bot": "私はAIなので特別な活動はしていませんが、多くの方とお話しす
{"user": "天気はどうですか？", "bot": "申し訳ありませんが、リアルタイムの天気情報は取得できません。お住ま
{"user": "時間を教えて", "bot": "申し訳ありませんが、現在時刻は表示できません。デバイスの時計をご確認くだ

{"user": "おすすめの映画を教えて", "bot": "映画のジャンルによりますね。アクション、ドラマ、コメディ、アニ
{"user": "音楽は好きですか？", "bot": "音楽について話すのは楽しいです！クラシック、ポップ、ロック、ジャズ
{"user": "読書について", "bot": "読書は知識を広げる素晴らしい趣味ですね。小説、ビジネス書、自己啓発書など

{"user": "勉強について相談があります", "bot": "勉強についてお手伝いできることがあれば嬉しいです。どのよう
{"user": "プログラミングを学びたい", "bot": "プログラミング学習は素晴らしい挑戦ですね！Python、JavaScript、
{"user": "仕事が大変です", "bot": "仕事でお疲れのようですね。具体的にどのような点が大変でしょうか？時間管

{"user": "ありがとうございます", "bot": "どういたしまして！お役に立てて嬉しいです。他にも何かご質問やお話
{"user": "また今度話しましょう", "bot": "はい、ぜひまたお話しましょう！いつでもお待ちしています。素晴らし
{"user": "さようなら", "bot": "さようなら！今日はお話しできて楽しかったです。また次回お会いできることを楽

{"user": "よくわからない質問", "bot": "申し訳ありませんが、質問の内容をもう少し具体的に教えていただけます
]
return conversations

conversations = create_comprehensive_dataset()

print(f"会話データセット作成完了: {len(conversations)}件の会話ペア")

print("\n=== データセット例 ===")
for i, conv in enumerate(conversations[:3]):
print(f"例 {i+1}:")
print(f" ユーザー: {conv['user']}")
print(f" ボット: {conv['bot']}")
print()

def load_conversation_data(file_path: str) -> List[Dict[str, str]]:
"""
JSONファイルから会話データを読み込む関数
JSONファイル形式例:
[
{"user": "質問1", "bot": "回答1"},
{"user": "質問2", "bot": "回答2"}
]
"""

try:
with open(file_path, 'r', encoding='utf-8') as f:
data = json.load(f)

if not isinstance(data, list):
raise ValueError("データは配列形式である必要があります")
for item in data:
if not isinstance(item, dict) or 'user' not in item or 'bot' not in item:
raise ValueError("各要素にはuserとbotキーが必要です")
print(f"外部ファイルから {len(data)} 件の会話データを読み込みました")
return data

except FileNotFoundError:
print(f"ファイル '{file_path}' が見つかりません。デフォルトデータを使用します。")
return create_comprehensive_dataset()
except json.JSONDecodeError as e:
print(f"JSONファイルの解析エラー: {e}")
return create_comprehensive_dataset()
except Exception as e:
print(f"データ読み込みエラー: {e}")
return create_comprehensive_dataset()

conversations = load_conversation_data('chatbot_data.json')

