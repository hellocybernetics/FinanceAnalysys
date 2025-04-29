import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

def fetch_stock_data(symbol, period='1y', interval='1d'):
    """
    株価データと財務データを取得する関数
    
    Args:
        symbol (str): 銘柄コード
        period (str): 取得期間
        interval (str): データ間隔
        
    Returns:
        tuple: (株価データ, 財務データのタプル)
    """
    try:
        # APIからデータ取得
        logger.info(f"Fetching {symbol} data from API")
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        
        # インデックスがDatetimeIndexであることを確認し、タイムゾーン情報を処理
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # タイムゾーン情報を処理
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        
        # 財務データの取得
        try:
            info = stock.info
            logger.info(f"Info data keys for {symbol}: {list(info.keys()) if info else 'No info data'}")
            
            quarterly_financials = stock.quarterly_financials
            logger.info(f"Quarterly financials for {symbol}: {quarterly_financials.index.tolist() if quarterly_financials is not None else 'No quarterly data'}")
            
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow
            
            fundamental_data = {
                'info': info,
                'quarterly_financials': quarterly_financials,
                'quarterly_balance_sheet': quarterly_balance_sheet,
                'quarterly_cashflow': quarterly_cashflow
            }
            logger.info(f"Successfully fetched fundamental data for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to fetch fundamental data for {symbol}: {str(e)}")
            fundamental_data = None
        
        return df, fundamental_data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None