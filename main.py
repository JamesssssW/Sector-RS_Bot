import os
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# 定義 20 個美股板塊及重點行業 ETF
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
    'XLC': '通訊服務',
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

def calculate_rrg_and_rs():
    print("正在下載數據並計算 RRG 四象限及回報率...")
    # 需要下載較長歷史數據（6個月）以計算平滑均線
    data = yf.download(tickers, period='6mo', progress=False)['Close']
    
    if isinstance(data, pd.Series):
        raise ValueError("下載數據出錯，請檢查網絡或 Ticker")

    # 1. 計算相對強度比值 (Ratio = Sector / SPY)
    ratios = pd.DataFrame()
    for t in SECTORS.keys():
        if t in data.columns:
            ratios[t] = data[t] / data[BENCHMARK]

    # 2. 計算 RRG 核心指標 (RS-Ratio & RS-Momentum)
    rs_ratio = pd.DataFrame()
    rs_momentum = pd.DataFrame()

    for t in SECTORS.keys():
        if t in ratios.columns:
            series = ratios[t]
            # 標準 RRG 邏輯：以 10日 與 50日 移動平均比率作為 RS-Ratio 基準 (中心值 100)
            sma_short = series.rolling(window=10).mean()
            sma_long = series.rolling(window=50).mean()
            rs_ratio[t] = 100 * (sma_short / sma_long)
            
            # RS-Momentum：RS-Ratio 的近期動能變化率
            mom_short = rs_ratio[t].rolling(window=5).mean()
            mom_long = rs_ratio[t].rolling(window=20).mean()
            rs_momentum[t] = 100 * (mom_short / mom_long)

    # 取得最新一天的數值
    latest_ratio = rs_ratio.iloc[-1]
    latest_mom = rs_momentum.iloc[-1]

    # 各週期超額回報計算 (1W=5d, 2W=10d, 1M=21d)
    days_1w, days_2w, days_1m = 5, 10, 21
    p_end = data.iloc[-1]
    
    spy_ret_1w = (p_end[BENCHMARK] / data.iloc[-days_1w][BENCHMARK]) - 1
    spy_ret_2w = (p_end[BENCHMARK] / data.iloc[-days_2w][BENCHMARK]) - 1
    spy_ret_1m = (p_end[BENCHMARK] / data.iloc[-days_1m][BENCHMARK]) - 1

    results = []
    for t, name in SECTORS.items():
        if t in data.columns:
            e1w = ((p_end[t] / data.iloc[-days_1w][t]) - 1 - spy_ret_1w) * 100
            e2w = ((p_end[t] / data.iloc[-days_2w][t]) - 1 - spy_ret_2w) * 100
            e1m = ((p_end[t] / data.iloc[-days_1m][t]) - 1 - spy_ret_1m) * 100
            
            r_ratio = float(latest_ratio.get(t, 100.0))
            r_mom = float(latest_mom.get(t, 100.0))

            # 四象限判斷邏輯
            if r_ratio >= 100 and r_mom >= 100:
                quadrant = "Leading"
                quad_emoji = "🟢"
            elif r_ratio < 100 and r_mom >= 100:
                quadrant = "Improving"
                quad_emoji = "🔵"
            elif r_ratio >= 100 and r_mom < 100:
                quadrant = "Weakening"
                quad_emoji = "🟡"
            else:
                quadrant = "Lagging"
                quad_emoji = "🔴"

            results.append({
                'ticker': t,
                'name': name,
                'excess_1w': e1w,
                'excess_2w': e2w,
                'excess_1m': e1m,
                'rs_ratio': r_ratio,
                'rs_momentum': r_mom,
                'quadrant': quadrant,
                'quad_emoji': quad_emoji
            })

    # 綜合排名主要依據 1 個月超額回報排序
    df_rs = pd.DataFrame(results)
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

    # 決定 Embed 主色 (根據第一名板塊的象限動態變色)
    top_quadrant = df.iloc[0]['quadrant'] if not df.empty else "Leading"
    color_map = {
        "Leading": 3066993,     # 🟢 綠色 (#2ECC71)
        "Improving": 3447003,   # 🔵 藍色 (#3498DB)
        "Weakening": 15844367,  # 🟡 黃色 (#F1C40F)
        "Lagging": 15158332     # 🔴 紅色 (#E74C3C)
    }
    embed_color = color_map.get(top_quadrant, 3447003)

    # 建立三大區塊的純文字表格
    # 1. 綜合排名（含 emoji + 象限）
    table_1 = f"{'名':<3}{'變':<4}{'ETF':<6}{'板塊名稱':<8}{'RRG 象限':<12}\n"
    table_1 += "-" * 38 + "\n"
    
    # 2. 各週期 RS 細節
    table_2 = f"{'ETF':<6}{'1W超額':<8}{'2W超額':<8}{'1M超額':<8}\n"
    table_2 += "-" * 32 + "\n"
    
    # 3. RRG 數值
    table_3 = f"{'ETF':<6}{'RS-Ratio':<10}{'RS-Momentum':<12}\n"
    table_3 += "-" * 30 + "\n"

    for idx, row in df.iterrows():
        rank = idx + 1
        ticker = row['ticker']
        prev_rank = prev_ranks.get(ticker)
        change_str = format_rank_change(rank, prev_rank)
        
        name = row['name']
        emoji = row['quad_emoji']
        quad = row['quadrant']
        
        table_1 += f"{rank:<3}{change_str:<4}{ticker:<6}{name:<8}{emoji} {quad:<8}\n"
        table_2 += f"{ticker:<6}{row['excess_1w']:+6.2f}% {row['excess_2w']:+6.2f}% {row['excess_1m']:+6.2f}%\n"
        table_3 += f"{ticker:<6}{row['rs_ratio']:<10.2f}{row['rs_momentum']:<12.2f}\n"

    # 組裝 Discord Embed
    embed = {
        "title": "📊 美股 20 大板塊 RRG 四象限與動能分析",
        "description": (
            f"**🏆 當前領頭羊象限：** `{top_quadrant}`\n\n"
            f"**【1. 綜合排名與象限】**\n```text\n{table_1}\n```\n"
            f"**【2. 各週期超額回報】**\n```text\n{table_2}\n```\n"
            f"**【3. RRG 核心數值 (基准100)】**\n```text\n{table_3}\n```"
        ),
        "color": embed_color,
        "footer": {
            "text": "GitHub Actions 自動發送 • RRG 四象限模型每日收盤更新"
        }
    }

    payload = {"embeds": [embed]}
    response = requests.post(webhook_url, json=payload)
    if response.status_code == 204:
        print("成功發送 RRG 報告到 Discord！")
    else:
        print(f"發送失敗，狀態碼：{response.status_code}, 回應：{response.text}")

if __name__ == "__main__":
    prev_ranks = load_previous_ranks()
    df_result = calculate_rrg_and_rs()
    send_to_discord(df_result, prev_ranks)
    save_current_ranks(df_result)
