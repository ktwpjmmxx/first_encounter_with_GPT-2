import streamlit as st
from unsloth import FastLanguageModel
import torch
import json
import os
from datetime import datetime

# ==========================================
# パス設定 (環境に合わせて修正してください)
# ==========================================
MODEL_PATH = "/content/drive/MyDrive/Llama3_FineTune/lora_model_llama3_final"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_DIR, 'assets') 

def get_asset_path(filename):
    """assetsフォルダ内のファイルの絶対パスを取得"""
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        return path
    return None

# ==========================================
# ページ設定
# ==========================================
st.set_page_config(
    page_title="Guardian AI - Legal Compliance (Local Ver.)",
    page_icon="🛡️", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# セッション状態の初期化
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'current_input' not in st.session_state:
    st.session_state.current_input = ""

# ==========================================
# CSSデザイン
# ==========================================
st.markdown("""
    <style>
    /* ベースフォント */
    .stApp {
        font-family: "Helvetica Neue", Arial, "Hiragino Kaku Gothic ProN", "Hiragino Sans", Meiryo, sans-serif;
    }

    /* サイドバー */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    /* サイドバー見出し */
    .sidebar-label {
        color: #475569;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 0.2rem;
    }

    /* ヘッダー・サブヘッダーのフォントサイズ統一 */
    .custom-header, .custom-subheader {
        font-size: 1.5rem !important; /* 強制的にサイズを統一 */
        font-weight: 700;
        color: #1e293b;
        margin: 0;
        padding-top: 8px; /* アイコンとの位置合わせ */
    }

    /* ボタン */
    div.stButton > button {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        color: white;
        border: 1px solid #0f172a;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(145deg, #334155 0%, #475569 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
        border-color: #475569;
    }

    /* サイドバーボタン */
    div[data-testid="stSidebar"] div.stButton > button {
        background: #ffffff;
        color: #334155;
        border: 1px solid #cbd5e1;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 0.9rem;
        padding: 0.5rem 0.8rem;
    }
    div[data-testid="stSidebar"] div.stButton > button:hover {
        background: #f1f5f9;
        border-color: #94a3b8;
        transform: none;
    }

    /* リスク判定結果ボックス */
    .risk-container {
        padding: 20px 24px;
        border-radius: 8px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        border: 1px solid #cbd5e1; 
        background-color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .risk-label {
        font-size: 1.0rem;
        font-weight: 700;
        color: #64748b;
        margin-right: 2rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .risk-value {
        font-size: 1.8rem;
        font-weight: 800;
        font-family: "Georgia", serif;
    }
    
    /* 色定義 */
    .color-High { color: #b91c1c; border-left: 6px solid #b91c1c; }
    .color-Medium { color: #b45309; border-left: 6px solid #b45309; }
    .color-Low { color: #047857; border-left: 6px solid #047857; }
    
    /* 関連法規タグ */
    .law-tag {
        display: inline-block;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0; 
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.95rem; /* 少し大きく */
        font-weight: 600;
        margin: 0 6px 6px 0;
    }

    /* Streamlit標準のsubheaderのスタイルを上書きして統一 */
    h3 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
        padding-top: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# ヘルパー関数
# ==========================================

def render_icon_header(text, icon_filename):
    """アイコン付きヘッダーを表示（サイズ統一）"""
    full_path = get_asset_path(icon_filename)
    
    # CSSクラスはすべて custom-header で統一
    text_class = "custom-header"

    if not full_path:
        st.markdown(f'<h3 style="padding-top:0;">{text}</h3>', unsafe_allow_html=True)
        return

    col_icon, col_text = st.columns([1.5, 10])

    with col_icon:
        st.image(full_path, use_container_width=True) 

    with col_text:
        # Pタグではなくdivで文字サイズをCSSで制御
        st.markdown(f'<div class="{text_class}">{text}</div>', unsafe_allow_html=True)

def render_sidebar_label(text, icon=""):
    st.markdown(f'<div class="sidebar-label">{icon} {text}</div>', unsafe_allow_html=True)

# ==========================================
# AIモデル設定 (Llama-3 Local)
# ==========================================

@st.cache_resource
def load_local_model():
    print(f"Loading Model from: {MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

try:
    with st.spinner('Guardian AI (Local Core) を起動中...'):
        model, tokenizer = load_local_model()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました。\nパス: {MODEL_PATH}\nエラー: {e}")
    st.stop()

def call_local_model(input_text):
    system_prompt = "IT法務コンサルタントとして回答してください。"
    prompt = f"""<|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

    outputs = model.generate(
        **inputs, 
        max_new_tokens = 512, 
        use_cache = True,
        temperature = 0.1,
    )
    result_text = tokenizer.batch_decode(outputs)[0]
    return result_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace("<|eot_id|>", "").strip()

def parse_model_output(raw_text):
    try:
        data = json.loads(raw_text)
        return {
            "risk_level": data.get("リスクレベル", "Medium"),
            "laws": [data.get("該当法", "不明")],
            "reason": data.get("理由", "詳細な理由を取得できませんでした。"),
            "recommendations": [data.get("修正案", "修正案を取得できませんでした。")]
        }
    except json.JSONDecodeError:
        return {
            "risk_level": "Check",
            "laws": ["-"],
            "reason": raw_text,
            "recommendations": ["-"]
        }

# ==========================================
# 結果表示ロジック
# ==========================================

def render_result(result_dict):
    if not result_dict: return

    st.markdown("---")
    
    risk = result_dict.get('risk_level', 'Medium')
    color_risk = risk if risk in ["High", "Medium", "Low"] else "Medium"
    
    html = f"""
    <div class="risk-container color-{color_risk}">
        <div class="risk-label">RISK ASSESSMENT</div>
        <div class="risk-value">{risk}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # 関連法規
    render_icon_header("関連法規", "icon_laws.png")
    laws = result_dict.get('laws', [])
    if isinstance(laws, str): laws = [laws]
    laws_html = "".join([f'<span class="law-tag">{law}</span>' for law in laws])
    st.markdown(laws_html, unsafe_allow_html=True)
    
    st.markdown("") 

    # リスク分析
    render_icon_header("リスク分析", "icon_analysis.png")
    st.write(result_dict.get('reason'))
    
    st.markdown("") 
    
    # 修正案（Recommendationの表記を削除し、ヘッダーサイズを他と統一）
    st.markdown('<h3 style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">💡 修正案</h3>', unsafe_allow_html=True)
    
    recs = result_dict.get('recommendations', [])
    if isinstance(recs, str): recs = [recs]
    for rec in recs:
        st.info(rec)

# ==========================================
# メインUI構築
# ==========================================

with st.sidebar:
    logo_path = get_asset_path("logo.png")
    if logo_path:
        st.image(logo_path, width=280) 
    else:
        st.markdown("## 🛡️ Guardian AI")

    render_sidebar_label("Quick Demo", "⚡")
    if st.button("事例: 偽装請負 (SES)"):
        st.session_state.current_input = "SESのエンジニアに対し、チャットで直接「明日は9時に来て」と指示を出したいです。効率のためです。"
        st.session_state.current_result = None 
        st.rerun()
    
    if st.button("事例: 下請法 (減額)"):
        st.session_state.current_input = "納品後のシステム代金、売上が悪いので10%減額で合意しました。問題ないですよね？"
        st.session_state.current_result = None 
        st.rerun()
        
    if st.button("事例: 雑談"):
        st.session_state.current_input = "最近腰が痛いんだけど、何かいいストレッチある？"
        st.session_state.current_result = None
        st.rerun()
            
    render_sidebar_label("Legend", "📊")
    st.caption("🔴 High: 重大な法的リスク")
    st.caption("🟠 Medium: 注意・要確認")
    st.caption("🟢 Low: リスク低")
    
    render_sidebar_label("History", "🕒")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            risk_val = item['result'].get('risk_level', 'Medium')
            risk_mark = "🔴" if risk_val == "High" else "🟠" if risk_val == "Medium" else "🟢"
            label = f"{risk_mark} {item.get('summary', '診断結果')}"
            if st.button(label, key=f"hist_{i}"):
                st.session_state.current_result = item['result']
                st.session_state.current_input = item['input']
                st.rerun()
    else:
        st.caption("履歴なし")
        
    st.markdown("---")
    if st.button("🗑️ 履歴クリア"):
        st.session_state.history = []
        st.session_state.current_result = None
        st.session_state.current_input = ""
        st.rerun()

# 修正箇所: タイトルを日本語に変更し、サイズはCSSで統一
render_icon_header("新規診断", "icon_new.png")

user_input = st.text_area(
    "仕様・サービス内容を入力してください", 
    value=st.session_state.current_input,
    height=150, 
    placeholder="例: ユーザーの購入履歴を分析し、本人同意なしで第三者に提供する機能を実装予定..."
)

if user_input != st.session_state.current_input:
    st.session_state.current_input = user_input

if st.button("リスク判定を実行する", type="primary"):
    if not user_input:
        st.warning("テキストを入力してください。")
    else:
        result_dict = None
        with st.spinner("Guardian AI (Llama-3) が推論中..."):
            try:
                raw_output = call_local_model(user_input)
                result_dict = parse_model_output(raw_output)
            except Exception as e:
                st.error(f"推論エラー: {e}")
        
        if result_dict:
            summary = user_input[:12] + "..."
            st.session_state.history.append({
                "input": user_input,
                "result": result_dict,
                "summary": summary,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.current_result = result_dict
            st.rerun()

if st.session_state.current_result:
    render_result(st.session_state.current_result)