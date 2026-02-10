"""
入力フィルタリングモジュール
対応範囲外のクエリを検出して、APIコストの無駄遣いを防ぎます。
"""

class InputFilter:
    """入力内容が対応範囲かどうかを判定するクラス"""
    
    def __init__(self):
        # 対応範囲外のキーワード
        self.out_of_scope_keywords = {
            "OSS": [
                "GPL", "MIT", "Apache", "BSD", "ライセンス違反",
                "オープンソース", "OSS", "LGPL", "MPL",
                "ソースコード公開", "再配布", "派生物"
            ],
            "AI倫理": [
                "AI倫理", "機械学習倫理", "バイアス", "公平性",
                "アルゴリズム差別", "透明性", "説明可能性",
                "AI偏見", "倫理的AI"
            ],
            "技術実装": [
                "SQL", "Python", "JavaScript", "React", "Vue",
                "サーバー構築", "AWS", "Azure", "GCP",
                "Docker", "Kubernetes", "API実装",
                "データベース設計", "セキュリティ実装",
                "暗号化アルゴリズム", "認証実装"
            ]
        }
    
    def check_scope(self, input_text: str) -> tuple[bool, str, str]:
        """
        入力テキストが対応範囲内かチェック
        
        Args:
            input_text: ユーザーの入力テキスト
            
        Returns:
            tuple: (is_in_scope, message, category)
                - is_in_scope: True=対応範囲内, False=対応範囲外
                - message: ユーザーへのメッセージ
                - category: 範囲外の場合のカテゴリ名
        """
        
        input_lower = input_text.lower()
        
        # 対応範囲外のキーワードチェック
        for category, keywords in self.out_of_scope_keywords.items():
            for keyword in keywords:
                if keyword.lower() in input_lower:
                    message = self._get_out_of_scope_message(category)
                    return False, message, category
        
        # すべてのチェックをパスした場合は対応範囲内
        return True, "", ""
    
    def _get_out_of_scope_message(self, category: str) -> str:
        """カテゴリに応じたメッセージを返す"""
        
        messages = {
            "OSS": """
本システムはOSSライセンスに関する診断には対応していません。
OSSライセンスの法的相談は、専門の法律事務所やライセンス互換性チェックツールの利用をご検討ください。
            """,
            "AI倫理": """
本システムはAI倫理に関する診断には対応していません。
バイアスや公平性の検証は、専用のフレームワークや倫理ガイドラインをご参照ください。
            """,
            "技術実装": """
本システムは技術的な実装コードの診断には対応していません。
セキュリティ実装やインフラ構築については、技術コンサルタントやセキュリティ診断をご検討ください。
            """
        }
        
        return messages.get(category, "対応範囲外の内容です。")