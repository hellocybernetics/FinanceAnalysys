import talib
import pandas as pd
import talib

def calculate_technical_indicators(df):
    """
    テクニカル指標を計算する関数
    
    Args:
        df (pd.DataFrame): 株価データ
        
    Returns:
        pd.DataFrame: テクニカル指標を追加したデータ
    """
    if df is None or df.empty:
        return None
        
    try:
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df['Close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # RSI
        df['RSI'] = talib.RSI(df['Close'])
        
        # ボリンジャーバンド
        upper, middle, lower = talib.BBANDS(df['Close'])
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        return df
    except Exception as e:
        return None
    

    