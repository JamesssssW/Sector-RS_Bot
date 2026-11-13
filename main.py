import os
import json
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
HISTORY_FILE = 'history.json'
tickers = list(SECTORS.keys()) + [BENCHMARK]

def load_previous_ranks():
    """讀取昨日儲存的排名紀錄"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"讀取歷史紀錄失敗：{e}")
    return {}

def save_current_ranks(df):
    """將今日最新排名寫入 history.json"""
    ranks = {row['ticker']: idx + 1 for idx, row in df.iterrows()}
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(ranks, f, ensure_ascii=False, indent=2)

def calculate_rs():
    print("正在下載板塊及大盤數據...")
    data = yf.download(tickers, period='6mo', progress=False)['Close']
    
    if isinstance(data, pd.Series):
        raise ValueError("下載數據出錯，請檢查網絡或 Ticker")

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
            excess_return = sector_ret - spy_ret
            rs_scores.append({
                'ticker': ticker,
                'name': name,
                'return': sector_ret * 100,
                'excess_return': excess_return * 100
            })
            
    df_rs = pd.DataFrame(rs_scores)
    df_rs = df_rs.sort_values(by='excess_return', ascending=False).reset_index(drop=True)
    return df_rs

def format_rank_change(current_rank, prev_rank):
    """計算名次變化並格式化顯示"""
    if prev_rank is None:
        return "NEW"
    diff = prev_rank - current_rank  # 昨日名次減今日名次，正數代表上升
    if diff > 0:
        return f"▲{diff}"
    elif diff < 0:
        return f"▼{abs(diff)}"
    else:
        return "─"

def send_to_discord(df, prev_ranks):
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        print("錯誤：未設定 DISCORD_WEBHOOK_URL 環境變數")
        return

    message = "📊 **美股板塊相對強度 (RS) 每日排行榜**\n"
    message += "基準：SPY | 周期：近 3 個月 (63日) 超額回報\n"
    message += "```text\n"
    message += f"{'排名':<4}{'變化':<6}{'ETF':<6}{'板塊名稱':<10}{'3M回報':<9}{'相對SPY超額':<10}\n"
    message += "-" * 52 + "\n"
    
    for idx, row in df.iterrows():
        rank = idx + 1
        ticker = row['ticker']
        prev_rank = prev_ranks.get(ticker)
        change_str = format_rank_change(rank, prev_rank)
        
        name = row['name']
        ret = f"{row['return']:+.2f}%"
        excess = f"{row['excess_return']:+.2f}%"
        message += f"{rank:<4}{change_str:<6}{ticker:<6}{name:<10}{ret:<9}{excess:<10}\n"
        
    message += "```"

    payload = {"content": message}
    response = requests.post(webhook_url, json=payload)
    if response.status_code == 204:
        print("成功發送訊息到 Discord！")
    else:
        print(f"發送失敗，狀態碼：{response.status_code}, 回應：{response.text}")

if __name__ == "__main__":
    prev_ranks = load_previous_ranks()
    df_result = calculate_rs()
    send_to_discord(df_result, prev_ranks)
    save_current_ranks(df_result)