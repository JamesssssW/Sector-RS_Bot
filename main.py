import os
import json
import requests
import yfinance as yf
import pandas as pd

# 定義 20 個美股板塊及重點行業 ETF
SECTORS = {
    # 11 大傳統 GICS 板塊
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
    'XLC': '通訊服務',
    # 9 大熱門重點行業 / 次板塊
    'SMH': '半導體',
    'XBI': '生物科技',
    'IGV': '軟體應用',
    'KRE': '地區銀行',
    'XHB': '房屋建築',
    'XRT': '零售',
    'IYT': '運輸',
    'ITA': '國防航天',
    'XME': '金屬礦業'
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
    print("正在下載 20 個板塊及大盤數據...")
    data = yf.download(tickers, period='3mo', progress=False)['Close']
    
    if isinstance(data, pd.Series):
        raise ValueError("下載數據出錯，請檢查網絡或 Ticker")

    days_1w = 5
    days_2w = 10
    days_1m = 21
    
    total_rows = len(data)
    if total_rows < days_1m:
        days_1m = total_rows - 1
    if total_rows < days_2w:
        days_2w = total_rows - 1
    if total_rows < days_1w:
        days_1w = total_rows - 1

    spy_ret_1w = (data[BENCHMARK].iloc[-1] / data[BENCHMARK].iloc[-days_1w]) - 1
    spy_ret_2w = (data[BENCHMARK].iloc[-1] / data[BENCHMARK].iloc[-days_2w]) - 1
    spy_ret_1m = (data[BENCHMARK].iloc[-1] / data[BENCHMARK].iloc[-days_1m]) - 1
    
    rs_scores = []
    for ticker, name in SECTORS.items():
        if ticker in data.columns:
            p_end = data[ticker].iloc[-1]
            
            ret_1w = (p_end / data[ticker].iloc[-days_1w]) - 1
            excess_1w = ret_1w - spy_ret_1w
            
            ret_2w = (p_end / data[ticker].iloc[-days_2w]) - 1
            excess_2w = ret_2w - spy_ret_2w
            
            ret_1m = (p_end / data[ticker].iloc[-days_1m]) - 1
            excess_1m = ret_1m - spy_ret_1m
            
            rs_scores.append({
                'ticker': ticker,
                'name': name,
                'excess_1w': excess_1w * 100,
                'excess_2w': excess_2w * 100,
                'excess_1m': excess_1m * 100
            })
            
    df_rs = pd.DataFrame(rs_scores)
    df_rs = df_rs.sort_values(by='excess_1m', ascending=False).reset_index(drop=True)
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

    # 組合 20 個板塊的表格內容
    table_content = f"{'排名':<4}{'變化':<6}{'ETF':<6}{'板塊名稱':<10}{'1W超額':<8}{'2W超額':<8}{'1M超額':<8}\n"
    table_content += "-" * 56 + "\n"
    
    for idx, row in df.iterrows():
        rank = idx + 1
        ticker = row['ticker']
        prev_rank = prev_ranks.get(ticker)
        change_str = format_rank_change(rank, prev_rank)
        
        name = row['name']
        exc_1w = f"{row['excess_1w']:+.2f}%"
        exc_2w = f"{row['excess_2w']:+.2f}%"
        exc_1m = f"{row['excess_1m']:+.2f}%"
        table_content += f"{rank:<4}{change_str:<6}{ticker:<6}{name:<10}{exc_1w:<8}{exc_2w:<8}{exc_1m:<8}\n"

    embed = {
        "title": "📊 美股 20 大板塊與行業相對強度排行榜",
        "description": f"基準：SPY | 比較周期：1週 / 2週 / 1個月 超額回報\n```text\n{table_content}\n```",
        "color": 3447003,
        "footer": {
            "text": "GitHub Actions 自動發送 • 每日美股收盤更新"
        }
    }

    payload = {"embeds": [embed]}
    response = requests.post(webhook_url, json=payload)
    if response.status_code == 204:
        print("成功發送 20 大板塊 Embed 卡片到 Discord！")
    else:
        print(f"發送失敗，狀態碼：{response.status_code}, 回應：{response.text}")

if __name__ == "__main__":
    prev_ranks = load_previous_ranks()
    df_result = calculate_rs()
    send_to_discord(df_result, prev_ranks)
    save_current_ranks(df_result)
