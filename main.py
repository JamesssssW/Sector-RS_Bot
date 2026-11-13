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
    # 前沿主題 / 賽道
    'CIBR': '網絡安全',
    'DRAM': '記憶體',
    'UFO': '太空探索',
    'LYTE': '光通信',
    'WGMI': '加密挖礦'
}

BENCHMARK = 'SPY'
HISTORY_FILE = 'history.json'
tickers = list(SECTORS.keys()) + [BENCHMARK]

def load_previous_history():
    """讀取歷史紀錄（包含過去的排名與象限序列）"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 相容舊版簡單格式
                if data and isinstance(list(data.values())[0], int):
                    new_data = {}
                    for k, v in data.items():
                        new_data[k] = {"ranks": [v], "quadrants": [None], "rank": v, "quadrant": None}
                    return new_data
                return data
        except Exception as e:
            print(f"讀取歷史紀錄失敗：{e}")
    return {}

def save_current_history(df_result):
    """將今日最新排名與象限追加寫入 history.json (最多保留最近 20 日)"""
    prev_data = load_previous_history()
    history_data = {}
    
    for idx, row in df_result.iterrows():
        ticker = row['ticker']
        curr_rank = idx + 1
        curr_quad = row['quadrant']
        
        item_hist = prev_data.get(ticker, {})
        ranks = item_hist.get('ranks', [])
        quadrants = item_hist.get('quadrants', [])
        
        ranks.append(curr_rank)
        quadrants.append(curr_quad)
        
        # 只保留最近 20 日數據
        ranks = ranks[-20:]
        quadrants = quadrants[-20:]
        
        history_data[ticker] = {
            "ranks": ranks,
            "quadrants": quadrants,
            "rank": curr_rank,
            "quadrant": curr_quad
        }
        
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

def calculate_rrg_and_rs():
    print("正在下載 25 個板塊數據並計算多週期動能與排名品質...")
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

    # 計算多週期回報率 (1D, 5D, 21D)
    ret_1d = data.pct_change(1).iloc[-1]
    ret_5d = data.pct_change(5).iloc[-1]
    ret_21d = data.pct_change(21).iloc[-1]

    spy_1d = ret_1d[BENCHMARK]
    spy_5d = ret_5d[BENCHMARK]
    spy_21d = ret_21d[BENCHMARK]

    results = []
    for t, name in SECTORS.items():
        if t in data.columns:
            e1d = (ret_1d[t] - spy_1d) * 100
            e5d = (ret_5d[t] - spy_5d) * 100
            e21d = (ret_21d[t] - spy_21d) * 100
            
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
                'rel_1d': e1d,
                'rel_5d': e5d,
                'rel_21d': e21d,
                'r_ratio': r_ratio,
                'r_mom': r_mom,
                'quadrant': quadrant,
                'quad_emoji': quad_emoji
            })

    df_rs = pd.DataFrame(results)
    # 初步按 1M 相對表現排序以得出今日暫定排名
    df_rs = df_rs.sort_values(by='rel_21d', ascending=False).reset_index(drop=True)
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

    # 整合表格：名(3) 變(4) ETF(6) 名稱(6) 象限(11) 1D相(6) 1W相(6) 1M相(6) 5D均(5) 品質/狀態(9)
    unified_table = f"{'名':<3}{'變':<4}{'ETF':<6}{'名稱':<6}{'象限':<11}{'1D相':<6}{'1W相':<6}{'1M相':<6}{'5D均':<5}{'品質/狀態':<9}\n"
    unified_table += "-" * 71 + "\n"

    for idx, row in df.iterrows():
        rank = idx + 1
        ticker = row['ticker']
        
        item_hist = prev_history.get(ticker, {})
        past_ranks = item_hist.get('ranks', [])
        past_quads = item_hist.get('quadrants', [])
        prev_rank = past_ranks[-1] if past_ranks else None
        prev_quad = past_quads[-1] if past_quads else None
        
        change_str = format_rank_change(rank, prev_rank)
        
        name = row['name']
        emoji = row['quad_emoji']
        quad = row['quadrant']

        # 計算歷史排名品質指標
        full_ranks = past_ranks + [rank]
        full_quads = past_quads + [quad]
        
        ranks_5d = full_ranks[-5:]
        avg_rank_5d = sum(ranks_5d) / len(ranks_5d) if ranks_5d else rank
        
        # 排名波動度（標準差）
        rank_std = np.std(ranks_5d) if len(ranks_5d) > 1 else 0.0
        
        # 計算 Leading 持續天數
        leading_days = 0
        for q in reversed(full_quads):
            if q == "Leading":
                leading_days += 1
            else:
                break

        # 品質與狀態判定
        if quad == "Leading":
            if leading_days <= 2:
                quality_str = "剛轉強⭐"
            else:
                quality_str = f"長期強({leading_days}天)"
        elif rank_std > 3.5:
            quality_str = "忽上忽下⚠️"
        elif avg_rank_5d < rank - 1:
            quality_str = "穩步上升📈"
        elif avg_rank_5d > rank + 1:
            quality_str = "回落中📉"
        else:
            quality_str = "橫盤穩定"

        has_rotated = False
        if prev_quad and prev_quad != quad:
            has_rotated = True
            prev_emoji = get_quadrant_emoji(prev_quad)
            rotations.append(f"• **{ticker} ({name})**: {prev_emoji} `{prev_quad}` ➔ {emoji} `{quad}`")

        r1d = f"{row['rel_1d']:+.1f}%"
        r5d = f"{row['rel_5d']:+.1f}%"
        r21d = f"{row['rel_21d']:+.1f}%"
        avg_5d_str = f"{avg_rank_5d:.1f}"

        if has_rotated:
            quad_display = f"{emoji}{quad[:6]}⭐"
        else:
            quad_display = f"{emoji}{quad:<9}"

        unified_table += f"{rank:<3}{change_str:<4}{ticker:<6}{name:<6}{quad_display:<11}{r1d:<6}{r5d:<6}{r21d:<6}{avg_5d_str:<5}{quality_str:<9}\n"

    # 📊 各象限分佈統計
    quad_counts = df['quadrant'].value_counts()
    count_leading = quad_counts.get("Leading", 0)
    count_improving = quad_counts.get("Improving", 0)
    count_weakening = quad_counts.get("Weakening", 0)
    count_lagging = quad_counts.get("Lagging", 0)

    market_breadth_summary = (
        f"**📈 市場板塊象限分佈統計 (共 {len(df)} 個標的)**\n"
        f"🟢 Leading: `{count_leading}` | 🔵 Improving: `{count_improving}` | "
        f"🟡 Weakening: `{count_weakening}` | 🔴 Lagging: `{count_lagging}`"
    )

    rotation_section = ""
    if rotations:
        rotation_section = "**🔄 RRG 跨象限轉倉快訊**\n" + "\n".join(rotations) + "\n\n"
    else:
        rotation_section = "**🔄 RRG 跨象限轉倉快訊**\n*今日暫無板塊發生跨象限轉倉.*\n\n"

    embed = {
        "title": "📊 美股 25 大板塊與主題品質監控系統",
        "description": (
            f"**🏆 當前領頭羊象限：** `{top_quadrant}`\n\n"
            f"{rotation_section}"
            f"**【板塊總覽與排名品質清單 (⭐ 代表轉倉/剛轉強)】**\n```text\n{unified_table}\n```\n"
            f"{market_breadth_summary}"
        ),
        "color": embed_color,
        "footer": {
            "text": "GitHub Actions 自動發送 • 多週期動能與排名品質模型"
        }
    }

    payload = {"embeds": [embed]}
    response = requests.post(webhook_url, json=payload)
    if response.status_code == 204:
        print("成功發送品質監控報告到 Discord！")
    else:
        print(f"發送失敗，狀態碼：{response.status_code}, 回應：{response.text}")

if __name__ == "__main__":
    prev_history = load_previous_history()
    df_result = calculate_rrg_and_rs()
    send_to_discord(df_result, prev_history)
    save_current_history(df_result)
