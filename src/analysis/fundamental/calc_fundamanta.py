from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)

def calculate_fundamental_indicators(fundamental_data, symbol):
    """
    ファンダメンタル指標を計算する関数
    
    Args:
        fundamental_data (dict): 財務データ
        symbol (str): 銘柄コード
        
    Returns:
        dict: 整形されたファンダメンタルデータ
    """
    if fundamental_data is None:
        return None
        
    try:
        # 財務データの展開
        info = fundamental_data['info']
        quarterly_financials = fundamental_data['quarterly_financials']
        quarterly_balance_sheet = fundamental_data['quarterly_balance_sheet']
        quarterly_cashflow = fundamental_data['quarterly_cashflow']
        
        # 市場と通貨の判定
        is_japanese = '.T' in symbol
        currency = info.get('financialCurrency', 'USD')
        market = 'JPX' if is_japanese else 'US'
        
        # 通貨調整係数（日本株の場合）
        currency_multiplier = 1.0/1000000.0 if currency == 'JPY' else 1.0
        
        # 基本データの構造化
        fundamental_dict = {
            "meta": {
                "symbol": symbol,
                "company_name": info.get('longName', ''),
                "market": market,
                "currency": currency,
                "last_update": datetime.now().isoformat()
            },
            "latest": {
                "valuation": {
                    "per": {"value": info.get('forwardPE', np.nan), "unit": "倍"},
                    "pbr": {"value": info.get('priceToBook', np.nan), "unit": "倍"},
                    "dividend_yield": {"value": info.get('dividendYield', np.nan) if info.get('dividendYield') else np.nan, "unit": "%"},
                    "ev_ebitda": {"value": info.get('enterpriseToEbitda', np.nan), "unit": "倍"}
                },
                "financial": {
                    "market_cap": {
                        "value": info.get('marketCap', np.nan) * currency_multiplier,
                        "unit": f"{currency}_M"
                    },
                    "shares_outstanding": {
                        "value": info.get('sharesOutstanding', np.nan),
                        "unit": "株"
                    }
                },
                "indicators": {
                    "roe": {
                        "value": info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
                        "unit": "%"
                    },
                    "roa": {
                        "value": info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') else np.nan,
                        "unit": "%"
                    },
                    "operating_margin": {
                        "value": info.get('operatingMargins', np.nan) * 100 if info.get('operatingMargins') else np.nan,
                        "unit": "%"
                    },
                    "net_margin": {
                        "value": info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
                        "unit": "%"
                    }
                },
                "growth": {
                    "revenue_growth": {
                        "value": info.get('revenueGrowth', np.nan) * 100 if info.get('revenueGrowth') else np.nan,
                        "unit": "%"
                    },
                    "earnings_growth": {
                        "value": info.get('earningsGrowth', np.nan) * 100 if info.get('earningsGrowth') else np.nan,
                        "unit": "%"
                    }
                }
            }
        }
        
        # 四半期データの処理
        if quarterly_financials is not None and not quarterly_financials.empty:
            try:
                dates = quarterly_financials.columns.strftime('%Y-%m-%d').tolist()
                quarterly_dict = {"dates": dates}
                
                # 売上高
                if 'Total Revenue' in quarterly_financials.index:
                    revenue = quarterly_financials.loc['Total Revenue'] * currency_multiplier
                    quarterly_dict["revenue"] = {
                        "values": revenue.tolist(),
                        "unit": f"{currency}_M"
                    }
                
                # 純利益
                if 'Net Income' in quarterly_financials.index:
                    net_income = quarterly_financials.loc['Net Income'] * currency_multiplier
                    quarterly_dict["net_income"] = {
                        "values": net_income.tolist(),
                        "unit": f"{currency}_M"
                    }
                
                # ROE
                if (quarterly_balance_sheet is not None and 
                    'Total Stockholder Equity' in quarterly_balance_sheet.index and
                    'Net Income' in quarterly_financials.index):
                    equity = quarterly_balance_sheet.loc['Total Stockholder Equity']
                    roe = (net_income / equity) * 100
                    quarterly_dict["roe"] = {
                        "values": roe.tolist(),
                        "unit": "%"
                    }
                
                fundamental_dict["quarterly"] = quarterly_dict
                
            except Exception as e:
                logger.error(f"Error processing quarterly data: {str(e)}")
        
        return fundamental_dict
        
    except Exception as e:
        logger.error(f"Error calculating fundamental indicators: {str(e)}")
        return None