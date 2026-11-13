import os
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# 定義 25 個美股板塊及重點主題 ETF
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
    'XME': '金屬礦業',
    # 前沿主題 / 賽道 (已更新 DRAM ETF、光通信 LYTE、加密挖礦 WGMI)
    'CIBR': '網絡安全',
    'DRAM': 'DRAM記憶體',
    'UFO': '太空探索',
    'LYTE': '光通信',
    'WGMI': '加密挖礦'
}

BENCHMARK = 'SPY'
HISTORY_FILE = 'history.json'
tickers = list(SECTORS.keys()) + [BENCHMARK]

def load_previous_history():
    """讀取昨日儲存的歷史紀錄（包含排名與象限）"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data and isinstance(list(data.values())[0], int):
                    return {k: {"rank": v, "quadrant": None} for k, v in data.items()}
                return data
        except Exception as e:
            print(f"讀取歷史紀錄失敗：{e}")
    return {}

def save_current_history(df):
    """將今日最新排名與象限寫入 history.json"""
    history_data = {}
    for idx, row in df.iterrows():
        history_data[row['ticker']] = {
            "rank": idx + 1,
            "quadrant": row['quadrant']
        }
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

def calculate_rrg_and_rs():
    print("正在下載 25 個板塊數據並計算 RRG 四象限及相對表現...")
    data = yf.download(tickers, period='6mo', progress=False)['Close']
    
    if isinstance(data, pd.Series):
        raise ValueError("下載數據出錯，請檢查網絡或 Ticker")

    ratios = pd.DataFrame()
    for t in SECTORS.keys():
        if t in data.columns:
            ratios[t] = data[t] / data[BENCHMARK]

    rs_ratio = pd.DataFrame()
    rs_momentum = pd.DataFrame()

    for t in SECTORS.keys():
        if t in ratios.columns:
            series = ratios[t]
            sma_short = series.rolling(window=10).mean()
            sma_long = series.rolling(window=50).mean()
            rs_ratio[t] = 100 * (sma_short / sma_long)
            
            mom_short = rs_ratio[t].rolling(window=5).mean()
            mom_long = rs_ratio[t].rolling(window=20).mean()
            rs_momentum[t] = 100 * (mom_short / mom_long)

    latest_ratio = rs_ratio.iloc[-1]
    latest_mom = rs_momentum.iloc[-1]

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
                'rel_1w': e1w,
                'rel_2w': e2w,
                'rel_1m': e1m,
                'rs_ratio': r_ratio,
                'rs_momentum': r_mom,
                'quadrant': quadrant,
                'quad_emoji': quad_emoji
            })

    df_rs = pd.DataFrame(results)
    df_rs = df_rs.sort_values(by='rel_1m', ascending=False).reset_index(drop=True)
    return df_rs

def format_rank_change(current_rank, prev_rank):
    if prev_rank is None:
        return "NEW"
    diff = prev_rank - current_rank
    if diff > 0:
        return f"▲{diff}"
    elif diff < 0:
        return f"▼{abs(diff)}"
    else:
        return "─"

def get_quadrant_emoji(quad):
    mapping = {
        "Leading": "🟢",
        "Improving": "🔵",
        "Weakening": "🟡",
        "Lagging": "🔴"
    }
    return mapping.get(quad, "⚪")

def send_to_discord(df, prev_history):
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        print("錯誤：未設定 DISCORD_WEBHOOK_URL 環境變數")
        return

    top_quadrant = df.iloc[0]['quadrant'] if not df.empty else "Leading"
    color_map = {
        "Leading": 3066993,     # 🟢 綠色
        "Improving": 3447003,   # 🔵 藍色
        "Weakening": 15844367,  # 🟡 黃色
        "Lagging": 15158332     # 🔴 紅色
    }
    embed_color = color_map.get(top_quadrant, 3447003)

    rotations = []

    unified_table = f"{'名':<3}{'變':<4}{'ETF':<6}{'名稱':<6}{'象限':<11}{'1W相':<7}{'2W相':<7}{'1M相':<7}{'Ratio':<7}{'Momentum':<8}\n"
    unified_table += "-" * 73 + "\n"

    for idx, row in df.iterrows():
        rank = idx + 1
        ticker = row['ticker']
        
        prev_info = prev_history.get(ticker, {})
        prev_rank = prev_info.get("rank")
        prev_quad = prev_info.get("quadrant")
        
        change_str = format_rank_change(rank, prev_rank)
        
        name = row['name']
        emoji = row['quad_emoji']
        quad = row['quadrant']

        if prev_quad and prev_quad != quad:
            prev_emoji = get_quadrant_emoji(prev_quad)
            rotations.append(f"• **{ticker} ({name})**: {prev_emoji} `{prev_quad}` ➔ {emoji} `{quad}`")

        r1w = f"{row['rel_1w']:+.1f}%"
        r2w = f"{row['rel_2w']:+.1f}%"
        r1m = f"{row['rel_1m']:+.1f}%"
        ratio_str = f"{row['rs_ratio']:.1f}"
        mom_str = f"{row['rs_momentum']:.1f}"

        if quad == "Leading":
            quad_display = "🟢Lead ⭐"
        else:
            quad_display = f"{emoji}{quad:<9}"

        unified_table += f"{rank:<3}{change_str:<4}{ticker:<6}{name:<6}{quad_display:<11}{r1w:<7}{r2w:<7}{r1m:<7}{ratio_str:<7}{mom_str:<8}\n"

    rotation_section = ""
    if rotations:
        rotation_section = "**🔄 RRG 跨象限轉倉快訊**\n" + "\n".join(rotations) + "\n\n"
    else:
        rotation_section = "**🔄 RRG 跨象限轉倉快訊**\n*今日暫無板塊發生跨象限轉倉.*\n\n"

    embed = {
        "title": "📊 美股 25 大板塊與主題綜合量化監控表",
        "description": (
            f"**🏆 當前領頭羊象限：** `{top_quadrant}`\n\n"
            f"{rotation_section}"
            f"**【板塊總覽與 RRG 動能清單 (⭐ 代表 Leading 領先區)】**\n```text\n{unified_table}\n```"
        ),
        "color": embed_color,
        "footer": {
            "text": "GitHub Actions 自動發送 • 綜合 RRG 與相對表現模型"
        }
    }

    payload = {"embeds": [embed]}
    response = requests.post(webhook_url, json=payload)
    if response.status_code == 204:
        print("成功發送綜合版塊報告到 Discord！")
    else:
        print(f"發送失敗，狀態碼：{response.status_code}, 回應：{response.text}")

if __name__ == "__main__":
    prev_history = load_previous_history()
    df_result = calculate_rrg_and_rs()
    send_to_discord(df_result, prev_history)
    save_current_history(df_result)
