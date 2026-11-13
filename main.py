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
    """讀取昨日儲存的 3M 排名紀錄"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"讀取歷史紀錄失敗：{e}")
    return {}

def save_current_ranks(df):
    """將今日最新 3M 排名寫入 history.json"""
    ranks = {row['ticker']: idx + 1 for idx, row in df.iterrows()}
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(ranks, f, ensure_ascii=False, indent=2)

def calculate_rs():
    print("正在下載板塊及大盤數據...")
    # 下載足夠覆蓋 3 個月（約 63 個交易日）的歷史數據
    data = yf.download(tickers, period='6mo', progress=False)['Close']
    
    if isinstance(data, pd.Series):
        raise ValueError("下載數據出錯，請檢查網絡或 Ticker")

    # 定義計算周期：1M = 21個交易日，3M = 63個交易日
    days_1m = 21
    days_3m = 63
    
    total_rows = len(data)
    if total_rows < days_3m:
        days_3m = total_rows - 1
    if total_rows < days_1m:
        days_1m = total_rows - 1

    # 計算基準 SPY 在不同周期的回報率
    spy_ret_3m = (data[BENCHMARK].iloc[-1] / data[BENCHMARK].iloc[-days_3m]) - 1
    spy_ret_1m = (data[BENCHMARK].iloc[-1] / data[BENCHMARK].iloc[-days_1m]) - 1
    
    rs_scores = []
    for ticker, name in SECTORS.items():
        if ticker in data.columns:
            # 3個月回報與超額回報
            p_start_3m = data[ticker].iloc[-days_3m]
            p_end = data[ticker].iloc[-1]
            ret_3m = (p_end / p_start_3m) - 1
            excess_3m = ret_3m - spy_ret_3m
            
            # 1個月回報與超額回報
            p_start_1m = data[ticker].iloc[-days_1m]
            ret_1m = (p_end / p_start_1m) - 1
            excess_1m = ret_1m - spy_ret_1m
            
            rs_scores.append({
                'ticker': ticker,
                'name': name,
                'return_3m': ret_3m * 100,
                'excess_3m': excess_3m * 100,
                'return_1m': ret_1m * 100,
                'excess_1m': excess_1m * 100
            })
            
    # 主要根據 3M 超額回報進行排名（可按需要改為 1M）
    df_rs = pd.DataFrame(rs_scores)
    df_rs = df_rs.sort_values(by='excess_3m', ascending=False).reset_index(drop=True)
    return df_rs

def format_rank_change(current_rank, prev_rank):
    """計算名次變化並格式化顯示"""
    if prev_rank is None:
        return "NEW"
    diff = prev_rank - current_rank
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

    # 1. 組合表格內容 (使用 Code Block 確保排版對齊)
    table_content = f"{'排名':<4}{'變化':<6}{'ETF':<6}{'板塊名稱':<10}{'1M超額':<9}{'3M超額':<9}\n"
    table_content += "-" * 50 + "\n"
    
    for idx, row in df.iterrows():
        rank = idx + 1
        ticker = row['ticker']
        prev_rank = prev_ranks.get(ticker)
        change_str = format_rank_change(rank, prev_rank)
        
        name = row['name']
        exc_1m = f"{row['excess_1m']:+.2f}%"
        exc_3m = f"{row['excess_3m']:+.2f}%"
        table_content += f"{rank:<4}{change_str:<6}{ticker:<6}{name:<10}{exc_1m:<9}{exc_3m:<9}\n"

    # 2. 建立 Discord Embed 卡片結構
    embed = {
        "title": "📊 美股板塊相對強度 (RS) 每日排行榜",
        "description": f"基準：SPY | 比較周期：近 1M (21日) & 近 3M (63日) 超額回報\n```text\n{table_content}\n```",
        "color": 3447003,  # 顏色邊框：這裡使用精緻的科技藍色 (十進位色碼，可自行更改)
        "footer": {
            "text": "GitHub Actions 自動發送 • 每日美股收盤更新"
        }
    }

    payload = {
        "embeds": [embed]
    }
    
    response = requests.post(webhook_url, json=payload)
    if response.status_code == 204:
        print("成功發送 Embed 卡片到 Discord！")
    else:
        print(f"發送失敗，狀態碼：{response.status_code}, 回應：{response.text}")

if __name__ == "__main__":
    prev_ranks = load_previous_ranks()
    df_result = calculate_rs()
    send_to_discord(df_result, prev_ranks)
    save_current_ranks(df_result)
