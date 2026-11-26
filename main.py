import os
import json
from datetime import datetime, timezone
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 1. 監控標的設定 (共 35 檔 ETF)
# ==========================================
CORE_SECTORS = {
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

THEMATIC_SECTORS = {
    'CIBR': '網絡安全',
    'DRAM': '記憶體',
    'UFO': '太空探索',
    'LYTE': '光通信',
    'WGMI': '加密挖礦',
    'ICLN': '清潔能源',
    'TAN': '太陽能',
    'BOTZ': '機械人AI',
    'SOCL': '社交媒體',
    'FINX': '金融科技',
    'COPX': '銅礦金屬',
    'IHI': '醫療設備',
    'SKYY': '雲端運算',
    'PAVE': '基礎設施',
    'JETS': '航空旅遊'
}

SECTORS = {**CORE_SECTORS, **THEMATIC_SECTORS}
BENCHMARK = 'SPY'
HISTORY_FILE = 'history.json'
tickers = list(SECTORS.keys()) + [BENCHMARK]


# ==========================================
# 2. 歷史紀錄處理 (保留最近 30 日)
# ==========================================
def load_previous_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
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

        ranks = ranks[-30:]
        quadrants = quadrants[-30:]

        history_data[ticker] = {
            "ranks": ranks,
            "quadrants": quadrants,
            "rank": curr_rank,
            "quadrant": curr_quad
        }

    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)


# ==========================================
# 3. 數據下載、大市環境與多週期加權計算
# ==========================================
def calculate_rrg_and_rs():
    print(f"正在下載 {len(SECTORS)} 個板塊 1 年數據並計算大市環境與加權動能...")

    try:
        # 修正：移除自訂 requests.Session，避免與新版 yfinance (curl_cffi) 衝突
        raw = yf.download(tickers, period='1y', progress=False, auto_adjust=True)
        data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    except Exception as e:
        raise RuntimeError(f"Yahoo Finance 數據下載失敗: {e}")

    if data is None or isinstance(data, pd.Series) or data.empty:
        raise ValueError("❌ 錯誤：Yahoo Finance 下載返回的數據為空，請稍後再試。")

    # 清除全部為 NaN 的無效欄位
    data = data.dropna(axis=1, how='all').ffill()

    if BENCHMARK not in data.columns or len(data) < 200:
        raise ValueError(f"❌ 錯誤：下載的 {BENCHMARK} 數據不足 200 筆（目前只有 {len(data)} 筆），無法計算 SMA200。")

    # 1. 計算 SPY 大市環境 (SMA200)
    spy_series = data[BENCHMARK].dropna()
    spy_sma200 = spy_series.rolling(window=200).mean().iloc[-1]
    spy_latest = spy_series.iloc[-1]

    if spy_latest >= spy_sma200:
        market_regime = "🟢 大市環境：多頭牛市 (SPY 企穩 SMA200)"
    else:
        market_regime = "🔴 大市環境：空頭警戒 (SPY 跌破 SMA200)"

    # 2. 計算 RRG
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

    # 3. 計算多週期相對回報率：1D (1), 5D (5), 2W (10), 1M (21)
    ret_1d = data.pct_change(1).iloc[-1]
    ret_5d = data.pct_change(5).iloc[-1]
    ret_2w = data.pct_change(10).iloc[-1]
    ret_1m = data.pct_change(21).iloc[-1]

    spy_1d = ret_1d[BENCHMARK]
    spy_5d = ret_5d[BENCHMARK]
    spy_2w = ret_2w[BENCHMARK]
    spy_1m = ret_1m[BENCHMARK]

    results = []
    for t, name in SECTORS.items():
        if t in data.columns:
            e1d = float((ret_1d[t] - spy_1d) * 100) if pd.notna(ret_1d[t]) else 0.0
            e5d = float((ret_5d[t] - spy_5d) * 100) if pd.notna(ret_5d[t]) else 0.0
            e2w = float((ret_2w[t] - spy_2w) * 100) if pd.notna(ret_2w[t]) else 0.0
            e1m = float((ret_1m[t] - spy_1m) * 100) if pd.notna(ret_1m[t]) else 0.0

            weighted_score = (e1d * 0.10) + (e5d * 0.30) + (e2w * 0.30) + (e1m * 0.30)

            r_ratio = float(latest_ratio.get(t, 100.0))
            r_mom = float(latest_mom.get(t, 100.0))
            if pd.isna(r_ratio):
                r_ratio = 100.0
            if pd.isna(r_mom):
                r_mom = 100.0

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
                'group': 'core' if t in CORE_SECTORS else 'thematic',
                'rel_1d': e1d,
                'rel_5d': e5d,
                'rel_2w': e2w,
                'rel_1m': e1m,
                'weighted_score': weighted_score,
                'r_ratio': r_ratio,
                'r_mom': r_mom,
                'quadrant': quadrant,
                'quad_emoji': quad_emoji
            })

    df_rs = pd.DataFrame(results)
    df_rs = df_rs.sort_values(by='weighted_score', ascending=False).reset_index(drop=True)
    return df_rs, market_regime


# ==========================================
# 4. 表格格式化與 Discord 雙 Embed 發送
# ==========================================
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


def generate_table_string(sub_df, prev_history):
    table_str = f"{'名':<3}{'變':<4}{'ETF':<6}{'名稱':<8}{'象限':<11}{'1D相':<6}{'5D相':<6}{'2W相':<6}{'1M相':<6}{'5D均':<5}{'品質/狀態':<9}\n"
    table_str += "-" * 79 + "\n\n"

    rotations = []

    for idx, row in sub_df.iterrows():
        rank = row['global_rank']
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

        full_ranks = past_ranks + [rank]
        full_quads = past_quads + [quad]

        ranks_5d = full_ranks[-5:]
        avg_rank_5d = sum(ranks_5d) / len(ranks_5d) if ranks_5d else rank
        rank_std = np.std(ranks_5d) if len(ranks_5d) > 1 else 0.0

        leading_days = 0
        for q in reversed(full_quads):
            if q == "Leading":
                leading_days += 1
            else:
                break

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
        r2w = f"{row['rel_2w']:+.1f}%"
        r1m = f"{row['rel_1m']:+.1f}%"
        avg_5d_str = f"{avg_rank_5d:.1f}"

        if has_rotated:
            quad_display = f"{emoji}{quad[:6]}⭐"
        else:
            quad_display = f"{emoji}{quad:<9}"

        table_str += f"{rank:<3}{change_str:<4}{ticker:<6}{name:<8}{quad_display:<11}{r1d:<6}{r5d:<6}{r2w:<6}{r1m:<6}{avg_5d_str:<5}{quality_str:<9}\n\n"

    return table_str, rotations


def send_to_discord(df, prev_history, market_regime):
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL', '').strip()
    if not webhook_url:
        raise RuntimeError("❌ 錯誤：未設定 DISCORD_WEBHOOK_URL 環境變數（或為空值）")

    df['global_rank'] = df.reset_index().index + 1

    df_core = df[df['group'] == 'core'].copy()
    df_thematic = df[df['group'] == 'thematic'].copy()

    table_core, rotations_core = generate_table_string(df_core, prev_history)
    table_thematic, rotations_thematic = generate_table_string(df_thematic, prev_history)
    all_rotations = rotations_core + rotations_thematic

    if all_rotations:
        rotation_section = "**🔄 RRG 跨象限轉倉快訊**\n" + "\n".join(all_rotations) + "\n\n"
    else:
        rotation_section = "**🔄 RRG 跨象限轉倉快訊**\n*今日暫無板塊發生跨象限轉倉.*\n\n"

    quad_counts = df['quadrant'].value_counts()
    market_breadth_summary = (
        f"**📈 市場板塊象限分佈統計 (共 {len(df)} 個標的)**\n"
        f"🟢 Leading: `{quad_counts.get('Leading', 0)}` | 🔵 Improving: `{quad_counts.get('Improving', 0)}` | "
        f"🟡 Weakening: `{quad_counts.get('Weakening', 0)}` | 🔴 Lagging: `{quad_counts.get('Lagging', 0)}`"
    )

    embed_core = {
        "title": "📊 美股傳統板塊與核心行業監控 (20 檔)",
        "description": (
            f"**{market_regime}**\n\n"
            f"{rotation_section}"
            f"**【核心資產總覽與排名品質】**\n```text\n{table_core}\n```"
        ),
        "color": 3066993
    }

    # 修正：使用帶 UTC 時區的 ISO 8601 格式
    current_time_iso = datetime.now(timezone.utc).isoformat()

    embed_thematic = {
        "title": "🚀 美股前沿主題與高增長賽道監控 (15 檔)",
        "description": (
            f"**【前沿主題總覽與排名品質】**\n```text\n{table_thematic}\n```\n"
            f"{market_breadth_summary}"
        ),
        "color": 3447003,
        "footer": {
            "text": "GitHub Actions 自動發送 • 加權多週期模型 (隔行優化版)"
        },
        "timestamp": current_time_iso
    }

    payload = {"embeds": [embed_core, embed_thematic]}
    response = requests.post(webhook_url, json=payload, timeout=30)

    if response.status_code in (200, 204):
        print("✅ 成功發送完整優化後的監控報告到 Discord！")
    else:
        # 修正：發送失敗時強制報錯，讓 GitHub Actions 顯示實際原因
        raise RuntimeError(f"❌ Discord 發送失敗，狀態碼：{response.status_code}, 回應：{response.text}")


# ==========================================
# 5. 主程式入口
# ==========================================
if __name__ == "__main__":
    prev_history = load_previous_history()
    df_result, market_regime = calculate_rrg_and_rs()
    send_to_discord(df_result, prev_history, market_regime)
    save_current_history(df_result)
