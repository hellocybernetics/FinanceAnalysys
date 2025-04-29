import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from matplotlib.gridspec import GridSpec
import numpy as np

def simple_technical_plot(df_plot, symbol):

    # 会社名の取得
    try:
        stock = yf.Ticker(symbol)
        company_name = stock.info.get('longName', symbol)
    except:
        company_name = symbol

    # 通貨単位の判定
    currency = 'JPY' if '.T' in symbol else 'USD'
    fig_tech = plt.figure(figsize=(15, 8))
    gs_tech = GridSpec(3, 1, figure=fig_tech, height_ratios=[1, 1, 1], hspace=0.2)
    
    # ボリンジャーバンドパネル
    ax1 = fig_tech.add_subplot(gs_tech[0])
    # バンド内を薄い赤で塗りつぶす
    ax1.fill_between(df_plot.index, df_plot['BB_Upper'], df_plot['BB_Lower'], color='red', alpha=0.1)
    ax1.plot(df_plot.index, df_plot['BB_Upper'], '--', color='red', label='Upper BB')
    ax1.plot(df_plot.index, df_plot['BB_Middle'], '-', color='green', label='Middle BB', linewidth=1.2)
    ax1.plot(df_plot.index, df_plot['BB_Lower'], '--', color='red', label='Lower BB')
    ax1.plot(df_plot.index, df_plot['Close'], '-', color='black', linewidth=0.75, label='Close')
    ax1.set_ylabel(f'Price ({currency})')
    ax1.grid(True, alpha=0.5, linestyle='dotted')
    ax1.legend(loc='upper left')
    ax1.yaxis.set_label_position('right')
    ax1.yaxis.tick_right()
    
    # MACDパネル
    ax2 = fig_tech.add_subplot(gs_tech[1])
    
    # MACDとシグナルライン（右軸）
    ax2_right = ax2.twinx()
    ax2_right.plot(df_plot.index, df_plot['MACD'], '-', color='#0066cc', label='MACD', linewidth=1.2)
    ax2_right.plot(df_plot.index, df_plot['MACD_Signal'], '-', color='#ff6600', label='Signal', linewidth=1.2)
    ax2_right.set_ylabel('MACD')
    ax2_right.legend(loc='upper right')
    
    # ヒストグラム（左軸）- 値に応じて色を変更
    pos_hist = df_plot['MACD_Hist'].copy()
    neg_hist = df_plot['MACD_Hist'].copy()
    pos_hist[pos_hist <= 0] = np.nan
    neg_hist[neg_hist > 0] = np.nan
    
    ax2.bar(df_plot.index, pos_hist, color='#26a69a', alpha=0.7, label='Histogram (Positive)')
    ax2.bar(df_plot.index, neg_hist, color='#ef5350', alpha=0.7, label='Histogram (Negative)')
    ax2.set_ylabel('MACD Histogram')
    ax2.legend(loc='upper left')
    
    # グリッドとスケールの設定
    ax2.grid(True, alpha=0.5, linestyle='dotted')
    
    # MACDとヒストグラムのY軸範囲を別々に設定
    # MACDのY軸範囲
    macd_min = min(df_plot['MACD'].min(), df_plot['MACD_Signal'].min())
    macd_max = max(df_plot['MACD'].max(), df_plot['MACD_Signal'].max())
    macd_margin = (macd_max - macd_min) * 0.1
    ax2_right.set_ylim(macd_min - macd_margin, macd_max + macd_margin)
    
    # ヒストグラムのY軸範囲
    hist_min = df_plot['MACD_Hist'].min()
    hist_max = df_plot['MACD_Hist'].max()
    hist_margin = (hist_max - hist_min) * 0.1
    ax2.set_ylim(hist_min - hist_margin, hist_max + hist_margin)
    
    # RSIパネル
    ax3 = fig_tech.add_subplot(gs_tech[2])
    ax3.plot(df_plot.index, df_plot['RSI'], '-', color='purple', label='RSI')
    ax3.axhline(y=70, color='red', linestyle='--', linewidth=0.5)
    ax3.axhline(y=30, color='red', linestyle='--', linewidth=0.5)
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.5, linestyle='dotted')
    ax3.legend(loc='upper left')
    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    
    # タイトルの設定（Figure 2）
    fig_tech.suptitle(f'{company_name}\n({symbol}) - Technical Indicators', y=0.95, fontsize=12)
    
    # レイアウトの調整（Figure 2）
    fig_tech.subplots_adjust(
        top=0.9,
        right=0.9,
        bottom=0.1
    )
    return fig_tech

def simple_price_plot(df_plot, symbol):
    # 通貨単位の判定
    currency = 'JPY' if '.T' in symbol else 'USD'
    
    # 会社名の取得
    try:
        stock = yf.Ticker(symbol)
        company_name = stock.info.get('longName', symbol)
    except:
        company_name = symbol
        
    # スタイル設定
    style = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        gridstyle='dotted',
        gridcolor='gray',
        gridaxis='both',
        y_on_right=True,
        marketcolors=mpf.make_marketcolors(
            up='forestgreen',
            down='red',
            edge='inherit',
            wick='inherit',
            volume='inherit'
        )
    )
    
    # Figure 1: ローソク足と出来高
    fig_price, axes_price = mpf.plot(
        df_plot,
        type='candle',
        style=style,
        volume=True,
        figsize=(15,7),
        title=f'\n\n{company_name} ({symbol})',
        ylabel=f'Price ({currency})',
        ylabel_lower='Volume',
        returnfig=True,
        tight_layout=False,
        panel_ratios=(3,1)
    )
    
    # タイトルの位置調整（Figure 1）
    fig_price.suptitle(f'{company_name}\n({symbol}) - Price & Volume', y=0.95, fontsize=12)
    
    # ボリュームの単位を調整（K/M/B）
    def volume_formatter(x, p):
        if x >= 1e9:
            return f'{x/1e9:.1f}B'
        elif x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        return f'{int(x)}'
    
    axes_price[1].yaxis.set_major_formatter(plt.FuncFormatter(volume_formatter))
    
    # Y軸を右側に配置（Figure 1）
    for ax in axes_price:
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.grid(True, alpha=0.5, linestyle='dotted')
    
    # レイアウトの調整（Figure 1）
    fig_price.subplots_adjust(
        top=0.9,
        right=0.9,
        bottom=0.1,
        hspace=0.1
    )
    return fig_price