def create_tweet_text(fundamental_path):
    """
    ファンダメンタルデータから120文字以内のツイートテキストを生成
    
    Args:
        fundamental_path (Path): ファンダメンタルデータのパス
        
    Returns:
        str: ツイートテキスト
    """
    try:
        with open(fundamental_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 企業名を取得
        company_name = None
        for line in lines:
            if line.startswith('企業名:'):
                company_name = line.split(':')[1].strip()
                break
        
        # 重要な指標を抽出
        indicators = {}
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('=== '):
                current_section = line.strip('= ')
            elif line and current_section:
                if current_section == 'バリュエーション指標':
                    if 'per:' in line.lower() or 'pbr:' in line.lower() or 'dividend_yield:' in line.lower():
                        key, value = line.split(':')
                        indicators[key.strip().lower()] = value.strip()
        
        # 指標の値を数値に変換（エラーハンドリング付き）
        try:
            per = float(indicators.get('per', '').split()[0]) if indicators.get('per') else None
        except (ValueError, IndexError):
            per = None
            
        try:
            pbr = float(indicators.get('pbr', '').split()[0]) if indicators.get('pbr') else None
        except (ValueError, IndexError):
            pbr = None
            
        try:
            dividend_yield = float(indicators.get('dividend_yield', '').split()[0]) if indicators.get('dividend_yield') else None
        except (ValueError, IndexError):
            dividend_yield = None
        
        # 市場動向コメントを生成
        per_comment = ""
        if per is not None:
            if per > 20:
                per_comment = "割高"
            elif per > 15:
                per_comment = "やや割高"
            elif per > 10:
                per_comment = "適正"
            else:
                per_comment = "割安"
            
        pbr_comment = ""
        if pbr is not None:
            if pbr > 2.0:
                pbr_comment = "割高"
            elif pbr > 1.5:
                pbr_comment = "やや割高"
            elif pbr > 1.0:
                pbr_comment = "適正"
            else:
                pbr_comment = "割安"
            
        dividend_comment = ""
        if dividend_yield is not None:
            if dividend_yield > 3.0:
                dividend_comment = "高配当"
            elif dividend_yield > 2.0:
                dividend_comment = "配当重視"
            else:
                dividend_comment = "成長重視"
        
        # ツイートテキストを生成
        text = f"{company_name}\n"
        if per is not None:
            text += f"PER: {per:.1f}倍({per_comment}) "
        if pbr is not None:
            text += f"PBR: {pbr:.1f}倍({pbr_comment}) "
        if dividend_yield is not None:
            text += f"配当利回り: {dividend_yield:.2f}%({dividend_comment})"
        
        # 120文字以内に収める
        if len(text) > 120:
            text = text[:117] + "..."
            
        return text
        
    except Exception as e:
        return None
