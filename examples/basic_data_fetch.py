import yfinance as yf
import pandas as pd
import numpy as np
import talib
import json
import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import mplfinance as mpf
import argparse
import sys
from pathlib import Path
from src.social.twitter.twitter_api_tweepy import TwitterAPITweepy
from src.analysis.fundamental.calc_fundamanta import calculate_fundamental_indicators
from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data
from src.visualization.simple_plot import simple_technical_plot, simple_price_plot
from src.social.twitter.create_tweet import create_tweet_text
from src.analysis.technical.calc_technical import calculate_technical_indicators

# Add project root to Python path
# プロジェクトルート（srcディレクトリの親）をsys.pathに追加
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_output_dir(symbol):
    """結果保存用のディレクトリを作成"""
    output_dir = Path('result') / 'basic_analysis'/ symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_fundamental_data(fundamental_dict, output_dir):
    """
    ファンダメンタルデータをテキストファイルとして保存
    
    Args:
        fundamental_dict (dict): ファンダメンタルデータの辞書
        output_dir (Path): 出力ディレクトリのパス
    """
    try:
        output_path = output_dir / 'ファンダメンタル.txt'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # 基本情報
            f.write("=== 基本情報 ===\n")
            f.write(f"企業名: {fundamental_dict['meta']['company_name']}\n")
            f.write(f"市場: {fundamental_dict['meta']['market']}\n")
            f.write(f"通貨: {fundamental_dict['meta']['currency']}\n\n")
            
            # バリュエーション指標
            f.write("=== バリュエーション指標 ===\n")
            for key, data in fundamental_dict['latest']['valuation'].items():
                f.write(f"{key}: {data['value']:.2f} {data['unit']}\n")
            f.write("\n")
            
            # 財務指標
            f.write("=== 財務指標 ===\n")
            for key, data in fundamental_dict['latest']['financial'].items():
                f.write(f"{key}: {data['value']:,.2f} {data['unit']}\n")
            f.write("\n")
            
            # 経営指標
            f.write("=== 経営指標 ===\n")
            for key, data in fundamental_dict['latest']['indicators'].items():
                f.write(f"{key}: {data['value']:.2f} {data['unit']}\n")
            f.write("\n")
            
            # 成長性指標
            f.write("=== 成長性指標 ===\n")
            for key, data in fundamental_dict['latest']['growth'].items():
                f.write(f"{key}: {data['value']:.2f} {data['unit']}\n")
            f.write("\n")
            
            # 四半期データ
            if 'quarterly' in fundamental_dict:
                f.write("=== 四半期データ ===\n")
                quarters = fundamental_dict['quarterly']['dates']
                
                if 'revenue' in fundamental_dict['quarterly']:
                    f.write("\n売上高:\n")
                    for date, value in zip(quarters, fundamental_dict['quarterly']['revenue']['values']):
                        f.write(f"{date}: {value:,.2f} {fundamental_dict['quarterly']['revenue']['unit']}\n")
                
                if 'net_income' in fundamental_dict['quarterly']:
                    f.write("\n純利益:\n")
                    for date, value in zip(quarters, fundamental_dict['quarterly']['net_income']['values']):
                        f.write(f"{date}: {value:,.2f} {fundamental_dict['quarterly']['net_income']['unit']}\n")
                
        logger.info(f"Saved fundamental data to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving fundamental data: {str(e)}")
        logger.exception("Detailed error information:")

def plot_stock_data(df, symbol, output_dir):
    """
    株価データをプロットして保存する関数
    
    Args:
        df (pd.DataFrame): 株価データとテクニカル指標
        symbol (str): 銘柄コード
        output_dir (Path): 出力ディレクトリのパス
    """
    if df is None or df.empty:
        logger.error(f"No data to plot for {symbol}")
        return
        
    try:

        # データの準備
        df_plot = df.copy()
        
        fig_price = simple_price_plot(df_plot, symbol)
        
        # ローソク足チャートの保存
        candle_path = output_dir / 'ローソク足.png'
        fig_price.savefig(candle_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved candlestick chart to {candle_path}")
        
        # Figure 2: テクニカル指標
        fig_tech = simple_technical_plot(df_plot, symbol)
        
        # テクニカル指標チャートの保存
        technical_path = output_dir / 'テクニカル.png'
        fig_tech.savefig(technical_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved technical chart to {technical_path}")
        
        # メモリ解放
        plt.close('all')
        
    except Exception as e:
        logger.error(f"Error plotting data for {symbol}: {str(e)}")
        logger.exception("Detailed error information:")


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description='株価データを取得し、テクニカル分析を行います。')
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='7203.T',
        help='銘柄コード（例: 7203.T）'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        default='2y',
        choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'],
        help='データ取得期間（デフォルト: 2y）'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        default='1d',
        choices=['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
        help='データ間隔（デフォルト: 1d）'
    )
    
    parser.add_argument(
        '--tweet',
        action='store_true',
        help='分析結果をツイートする'
    )

    parser.add_argument(
        '--comment',
        type=str,
        default='None',
        help='入れた文章がツイート時に上書きされます。'
    )

    return parser.parse_args()

def main():
    """メイン処理"""
    # コマンドライン引数の解析
    args = parse_args()
    
    # 出力ディレクトリの設定
    output_dir = setup_output_dir(args.symbol)
    
    # データ取得
    df, fundamental_data = fetch_stock_data(
        args.symbol,
        period=args.period,
        interval=args.interval
    )
    
    if df is None:
        logger.error(f"Failed to fetch data for {args.symbol}")
        return
    
    # テクニカル指標の計算
    df = calculate_technical_indicators(df)
    if df is None:
        logger.error(f"Failed to calculate technical indicators for {args.symbol}")
        return
        
    # ファンダメンタル指標の計算と保存
    if fundamental_data:
        logger.info(f"Calculating fundamental indicators for {args.symbol}")
        fundamental_dict = calculate_fundamental_indicators(fundamental_data, args.symbol)
        if fundamental_dict:
            save_fundamental_data(fundamental_dict, output_dir)
            logger.info(f"Saved fundamental data for {args.symbol}")
        else:
            logger.warning(f"No fundamental data to save for {args.symbol}")

    # プロットの生成と保存
    plot_stock_data(df, args.symbol, output_dir)
    
    # ツイート投稿
    if args.tweet:
        try:
            # ツイートテキストを生成
            if args.comment == 'None':
                fundamental_path = output_dir / 'ファンダメンタル.txt'
                tweet_text = create_tweet_text(fundamental_path)
            else:
                tweet_text = args.comment
            if not tweet_text:
                logger.error("ツイートテキストの生成に失敗しました")
                return            
            # 画像パスを準備
            image_paths = [
                str(output_dir / 'ローソク足.png'),
                str(output_dir / 'テクニカル.png')
            ]
            
            # 画像の存在確認
            for path in image_paths:
                if not Path(path).exists():
                    logger.error(f"画像ファイルが見つかりません: {path}")
                    return
            
            # TwitterAPIクライアントを初期化
            twitter_api = TwitterAPITweepy()
            
            # ツイート投稿
            tweet_id = twitter_api.post_tweet(tweet_text, image_paths)
            if tweet_id:
                logger.info(f"ツイートが投稿されました: https://twitter.com/user/status/{tweet_id}")
            else:
                logger.error("ツイートの投稿に失敗しました")
                
        except Exception as e:
            logger.error(f"ツイート処理中にエラーが発生しました: {e}")
    
    logger.info(f"処理が完了しました。結果は {output_dir} に保存されています。")

if __name__ == "__main__":
    main() 