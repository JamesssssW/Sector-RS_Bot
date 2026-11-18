import os
import json
from datetime import datetime
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

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
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    retries = Retry(
        total=5,
        backoff_factor=5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    
    original_request = session.request
    def request_with_timeout(*args, **kwargs):
        kwargs.setdefault('timeout', 30)
        return original_request(*args, **kwargs)
    session.request = request_with_timeout

    try:
        data = yf.download(tickers, period='1y', session=session, progress=False)['Close']
    except Exception as e:
        raise RuntimeError(f"Yahoo Finance 數據下載失敗或連線逾時: {e}")
    
    if data is None or isinstance(data, pd.Series) or data.empty:
        raise ValueError("❌ 錯誤：Yahoo Finance 下載返回的數據為空，可能被 Rate Limit (429) 封鎖，請稍後再試。")

    if BENCHMARK not in data.columns or len(data) < 200:
        raise ValueError(f"❌ 錯誤：下載的 {BENCHMARK} 數據不足 200 筆（目前只有 {len(data)} 筆），無法計算 SMA200。")

    # 🌐 1. 計算 SPY 大市環境 (SMA200)
    spy_series = data[BENCHMARK].dropna()
    spy_sma200 = spy_series.rolling(window=200).mean().iloc[-1]
    spy_latest = spy_series.iloc[-1]
    
    if spy_latest >= spy_sma200:
        market_regime = "🟢 大市環境：多頭牛市 (SPY 企穩 SMA200)"
    else:
        market_regime = "🔴 大市環境：空頭警戒 (SPY 跌破 SMA200)"

    # 🌐 2. 計算 RRG
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

    # 🌐 3. 計算多週期相對回報率：1D (1), 5D (5), 2W (10), 1M (21)
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
            e1d = (ret_1d[t] - spy_1d) * 100
            e5d = (ret_5d[t] - spy_5d) * 100
            e2w = (ret_2w[t] - spy_2w) * 100
            e1m = (ret_1m[t] - spy_1m) * 100
            
            # 短線加權綜合得分 (1D:10%, 5D:30%, 2W:30%, 1M:30%)
            weighted_score = (e1d * 0.10) + (e5d * 0.30) + (e2w * 0.30) + (e1m * 0.30)
            
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
