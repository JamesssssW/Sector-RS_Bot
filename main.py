import os
import requests
import yfinance as yf
import pandas as pd

# 定義美股 11 大板塊 ETF 及其中文名稱
SECTORS = {
    'XLK': '科技',
    'XLF': '金融',
    'XLV': '醫療保健',
    'XLY': '非必需消費',
    'XLP': '必需消費',
    'XLE': '能源',
    'XLI': '工業',
    'XLU': '公用事業',
    'XLB': '原材料',
    'XLRE': '房地產',
    'XLC': '通訊服務'
}

BENCHMARK = 'SPY'
tickers = list(SECTORS.keys()) + [BENCHMARK]

def calculate_rs():
    print("正在下載板塊及大盤數據...")
    # 下載過去 6 個月的收盤價
    data = yf.download(tickers, period='6mo', progress=False)['Close']
    
    if isinstance(data, pd.Series):
        raise ValueError("下載數據出錯，請檢查網絡或 Ticker")

    # 以 3 個月（約 63 個交易日）計算基準 SPY 的回報率
    lookback_days = 63
    if len(data) < lookback_days:
        lookback_days = len(data) - 1

    spy_ret = (data[BENCHMARK].iloc[-1] / data[BENCHMARK].iloc[-lookback_days]) - 1
    
    rs_scores = []
    for ticker, name in SECTORS.items():
        if ticker in data.columns:
            start_price = data[ticker].iloc[-lookback_days]
            end_price = data[ticker].iloc[-1]
            sector_ret = (end_price / start_price) - 1
            
            # 相對強度：板塊回報率減去 SPY 回報率（超額回報）
            excess_return = sector_ret - spy_ret
            rs_scores.append({
                'ticker': ticker,
                'name': name,
                'return': sector_ret * 100,
                'excess_return': excess_return * 100
            })
            
    # 按超額回報（強弱）由高到低排序
    df_rs = pd.DataFrame(rs_scores)
    df_rs = df_rs.sort_values(by='excess_return', ascending=False).reset_index(drop=True)
    return df_rs

def send_to_discord(df):
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        print("錯誤：未設定 DISCORD_WEBHOOK_URL 環境變數")
        return

    # 排版 Discord 訊息 (使用 Markdown Code Block 確保對齊)
    message = "📊 **美股板塊相對強度 (RS) 每日排行榜**\n"
    message += f"基準：SPY | 周期：近 3 個月 (63日) 超額回報\n"
    message += "```text\n"
    message += f"{'排名':<4}{'ETF':<6}{'板塊名稱':<10}{'3M回報':<9}{'相對SPY超額':<10}\n"
    message += "-" * 45 + "\n"
    
    for idx, row in df.iterrows():
        rank = idx + 1
        ticker = row['ticker']
        name = row['name']
        ret = f"{row['return']:+.2f}%"
        excess = f"{row['excess_return']:+.2f}%"
        message += f"{rank:<4}{ticker:<6}{name:<10}{ret:<9}{excess:<10}\n"
        
    message += "```"

    payload = {
        "content": message
    }
    
    response = requests.post(webhook_url, json=payload)
    if response.status_code == 204:
        print("成功發送訊息到 Discord！")
    else:
        print(f"發送失敗，狀態碼：{response.status_code}, 回應：{response.text}")

if __name__ == "__main__":
    df_result = calculate_rs()
    print(df_result)
    send_to_discord(df_result)