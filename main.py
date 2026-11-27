import os
import json
import time
from datetime import datetime, timezone
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 1. 全市場 60 檔輪動標的設定 (分三大矩陣，每組 20 檔)
# ==========================================
# 矩陣 1：11 大一級板塊 + 核心權重行業 (20 檔)
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

# 矩陣 2：前沿科技、AI 基建與細分賽道 (20 檔)
SUB_THEMATIC_SECTORS = {
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
    'IHI': '醫療設備',
    'SKYY': '雲端運算',
    'PAVE': '基礎設施',
    'JETS': '航空旅遊',
    'DTCR': '數據中心',
    'GRID': '智能電網',
    'QTUM': '量子計算',
    'FDN': '互聯網',
    'ESPO': '電競遊戲',
    'PPH': '大型製藥'
}

# 矩陣 3：大宗資源、傳統細分週期與跨資產宏觀輪動 (20 檔)
MACRO_COMMODITY_SECTORS = {
    'COPX': '銅礦金屬',
    'GDX': '金礦開採',
    'SIL': '銀礦開採',
    'URA': '鈾礦核電',
    'SLX': '鋼鐵產業',
    'REMX': '稀土戰略',
    'LIT': '鋰電儲能',
    'MOO': '農業食品',
    'PHO': '水資源',
    'XOP': '油氣開採',
    'OIH': '油田服務',
    'KIE': '保險行業',
    'IAI': '券商投行',
    'IHF': '醫療服務',
    'IWM': '羅素小盤',
    'GLD': '黃金現貨',
    'SLV': '白銀現貨',
    'TLT': '長期美債',
    'HYG': '高收益債',
    'IBIT': '比特幣'
}

SECTORS = {**CORE_SECTORS, **SUB_THEMATIC_SECTORS, **MACRO_COMMODITY_SECTORS}
BENCHMARK = 'SPY'
HISTORY_FILE = 'history.json'
tickers = list(SECTORS.keys()) + [BENCHMARK]

# 等風險倉位與 ATR 止損參數（可按個人風險胃納調整）
TARGET_POS_VOL = 3.0   # 單一倉位目標年化波動貢獻 (%)
MAX_POS_WEIGHT = 25.0  # 單一板塊最大倉位上限 (%)
ATR_PERIOD = 14        # ATR 計算週期（交易日）
ATR_MULTIPLIER = 2.0   # 動態追蹤止損倍數


# ==========================================
# 2. 歷史紀錄處理 (按美股交易日期去重，保留最近 30 個交易日)
# ==========================================
def load_previous_history():
    """讀取歷史紀錄並自動相容舊版格式，統一為 records 結構"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            normalized = {}
            for k, v in data.items():
                if isinstance(v, int):
                    normalized[k] = {
                        "records": [{"date": "legacy_0", "rank": v, "quadrant": None}],
                        "rank": v,
                        "quadrant": None
                    }
                elif isinstance(v, dict) and "records" in v:
                    normalized[k] = v
                elif isinstance(v, dict) and "ranks" in v:
                    ranks = v.get("ranks", [])
                    quads = v.get("quadrants", [None] * len(ranks))
                    records = [
                        {"date": f"legacy_{i}", "rank": r, "quadrant": q}
                        for i, (r, q) in enumerate(zip(ranks, quads))
                    ]
                    normalized[k] = {
                        "records": records[-30:],
                        "rank": v.get("rank", ranks[-1] if ranks else None),
                        "quadrant": v.get("quadrant", quads[-1] if quads else None)
                    }
            return normalized
        except Exception as e:
            print(f"讀取歷史紀錄失敗：{e}")
    return {}


def save_current_history(df_result, trade_date):
    """按美股交易日期寫入或覆蓋當日紀錄，嚴格保留最近 30 個交易日"""
    prev_data = load_previous_history()
    history_data = {}

    for idx, row in df_result.iterrows():
        ticker = row['ticker']
        curr_rank = int(idx + 1)
        curr_quad = row['quadrant']

        item_hist = prev_data.get(ticker, {})
        records = list(item_hist.get('records', []))

        new_entry = {
            "date": trade_date,
            "rank": curr_rank,
            "quadrant": curr_quad
        }

        if records and records[-1].get('date') == trade_date:
            records[-1] = new_entry
        else:
            records = [r for r in records if r.get('date') != trade_date]
            records.append(new_entry)

        records = records[-30:]

        history_data[ticker] = {
            "records": records,
            "rank": curr_rank,
            "quadrant": curr_quad
        }

    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)


# ==========================================
# 3. 數據下載、風險調整後動能、波動率倉位與 14 日 ATR 計算
# ==========================================
def get_group_name(ticker):
    if ticker in CORE_SECTORS:
        return 'core'
    elif ticker in SUB_THEMATIC_SECTORS:
        return 'sub_thematic'
    return 'macro_commodity'


def calculate_rrg_and_rs():
    print(f"正在下載全市場 {len(SECTORS)} 檔板塊與宏觀 ETF 1 年數據...")

    try:
        raw = yf.download(tickers, period='1y', progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            data = raw['Close']
            high_df = raw['High']
            low_df = raw['Low']
        else:
            data = raw
            high_df = raw
            low_df = raw
    except Exception as e:
        raise RuntimeError(f"Yahoo Finance 數據下載失敗: {e}")

    if data is None or isinstance(data, pd.Series) or data.empty:
        raise ValueError("❌ 錯誤：Yahoo Finance 下載返回的數據為空，請稍後再試。")

    data = data.dropna(axis=1, how='all').ffill()
    high_df = high_df.reindex(columns=data.columns).ffill()
    low_df = low_df.reindex(columns=data.columns).ffill()

    if BENCHMARK not in data.columns or len(data) < 200:
        raise ValueError(f"❌ 錯誤：下載的 {BENCHMARK} 數據不足 200 筆（目前只有 {len(data)} 筆），無法計算 SMA200。")

    # 1. 取得最新美股交易日期及 SPY 大市環境 (SMA200)
    spy_series = data[BENCHMARK].dropna()
    trade_date = spy_series.index[-1].strftime('%Y-%m-%d')
    spy_sma200 = spy_series.rolling(window=200).mean().iloc[-1]
    spy_latest = spy_series.iloc[-1]
    is_bull_market = bool(spy_latest >= spy_sma200)

    if is_bull_market:
        market_regime = f"🟢 大市環境：多頭牛市 (SPY 企穩 SMA200｜美股交易日：{trade_date})"
    else:
        market_regime = f"🔴 大市環境：空頭警戒 (SPY 跌破 SMA200｜美股交易日：{trade_date})"

    # 2. 計算 RRG (RS-Ratio & RS-Momentum)
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

    # 3. 計算多週期相對回報率與 20 日實現波動率
    daily_returns = data.pct_change(1, fill_method=None)
    vol_20d_daily = daily_returns.rolling(window=20).std().iloc[-1] * 100

    ret_1d = daily_returns.iloc[-1]
    ret_5d = data.pct_change(5, fill_method=None).iloc[-1]
    ret_2w = data.pct_change(10, fill_method=None).iloc[-1]
    ret_1m = data.pct_change(21, fill_method=None).iloc[-1]

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

            raw_vol = float(vol_20d_daily[t]) if t in vol_20d_daily.index and pd.notna(vol_20d_daily[t]) else 1.0
            safe_vol = max(raw_vol, 0.2)
            ann_vol_20d = safe_vol * np.sqrt(252)

            risk_adj_score = weighted_score / safe_vol
            pos_size = min((TARGET_POS_VOL / ann_vol_20d) * 100.0, MAX_POS_WEIGHT)

            # 14 日 Wilder ATR 與動態追蹤止損位
            close_s = data[t]
            high_s = high_df[t]
            low_s = low_df[t]
            prev_close_s = close_s.shift(1)

            tr = pd.concat([
                high_s - low_s,
                (high_s - prev_close_s).abs(),
                (low_s - prev_close_s).abs()
            ], axis=1).max(axis=1)

            atr_series = tr.ewm(alpha=1 / ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()
            latest_close = float(close_s.iloc[-1])
            atr_14 = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
            atr_stop_price = max(latest_close - (ATR_MULTIPLIER * atr_14), 0.0)
            atr_stop_pct = ((ATR_MULTIPLIER * atr_14) / latest_close * 100.0) if latest_close > 0 else 0.0

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
                'group': get_group_name(t),
                'close': latest_close,
                'atr_14': atr_14,
                'atr_stop_price': atr_stop_price,
                'atr_stop_pct': atr_stop_pct,
                'rel_1d': e1d,
                'rel_5d': e5d,
                'rel_2w': e2w,
                'rel_1m': e1m,
                'weighted_score': weighted_score,
                'vol_20d': safe_vol,
                'ann_vol_20d': ann_vol_20d,
                'pos_size': pos_size,
                'risk_adj_score': risk_adj_score,
                'r_ratio': r_ratio,
                'r_mom': r_mom,
                'quadrant': quadrant,
                'quad_emoji': quad_emoji
            })

    df_rs = pd.DataFrame(results)
    df_rs = df_rs.sort_values(by='risk_adj_score', ascending=False).reset_index(drop=True)
    return df_rs, market_regime, trade_date, is_bull_market


# ==========================================
# 4. 表格格式化與 Discord 四張獨立 Embed 分批發送
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


def generate_table_string(sub_df, prev_history, trade_date):
    table_str = (
        f"{'名':<3}{'變':<4}{'ETF':<6}{'名稱':<8}{'象限':<11}{'風調':<7}"
        f"{'年波':<5}{'單倉':<5}{'1D相':<6}{'5D相':<6}{'2W相':<6}{'1M相':<6}{'5D均':<5}{'品質/狀態':<9}\n"
    )
    table_str += "-" * 96 + "\n\n"

    rotations = []

    for idx, row in sub_df.iterrows():
        rank = int(row['global_rank'])
        ticker = row['ticker']

        item_hist = prev_history.get(ticker, {})
        all_records = item_hist.get('records', [])

        past_records = [r for r in all_records if r.get('date') != trade_date]
        past_ranks = [r['rank'] for r in past_records]
        past_quads = [r['quadrant'] for r in past_records]
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
        rank_std = float(np.std(ranks_5d)) if len(ranks_5d) > 1 else 0.0

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
        elif rank < avg_rank_5d - 1:
            quality_str = "穩步上升📈"
        elif rank > avg_rank_5d + 1:
            quality_str = "回落中📉"
        else:
            quality_str = "橫盤穩定"

        has_rotated = False
        if prev_quad and prev_quad != quad:
            has_rotated = True
            prev_emoji = get_quadrant_emoji(prev_quad)
            # 轉倉優先級：轉入 Leading(0) > 轉入 Improving(1) > 其他(2)
            priority = 0 if quad == "Leading" else (1 if quad == "Improving" else 2)
            rotations.append((priority, rank, f"• **#{rank} {ticker} ({name})**: {prev_emoji} `{prev_quad}` ➔ {emoji} `{quad}`"))

        radj = f"{row['risk_adj_score']:+.2f}"
        ann_vol = f"{row['ann_vol_20d']:.0f}%"
        pos_sz = f"{row['pos_size']:.0f}%"
        r1d = f"{row['rel_1d']:+.1f}%"
        r5d = f"{row['rel_5d']:+.1f}%"
        r2w = f"{row['rel_2w']:+.1f}%"
        r1m = f"{row['rel_1m']:+.1f}%"
        avg_5d_str = f"{avg_rank_5d:.1f}"

        if has_rotated:
            quad_display = f"{emoji}{quad[:6]}⭐"
        else:
            quad_display = f"{emoji}{quad:<9}"

        table_str += (
            f"{rank:<3}{change_str:<4}{ticker:<6}{name:<8}{quad_display:<11}{radj:<7}"
            f"{ann_vol:<5}{pos_sz:<5}{r1d:<6}{r5d:<6}{r2w:<6}{r1m:<6}{avg_5d_str:<5}{quality_str:<9}\n\n"
        )

    return table_str, rotations


def build_vol_sizing_summary(df, is_bull_market):
    """對全市場排名前 5 的 Leading 標的計算反波動率等風險權重與 14 日 ATR 止損位"""
    top_leading = df[df['quadrant'] == 'Leading'].head(5).copy()
    if top_leading.empty:
        return "**⚖️ Top 5 Leading 等風險配倉與止損參考**\n*目前暫無 Leading 象限標的，建議保留現金觀望。*"

    inv_vols = 1.0 / top_leading['ann_vol_20d']
    weights = (inv_vols / inv_vols.sum()) * 100.0
    exposure_scale = 1.0 if is_bull_market else 0.5

    lines = []
    for (_, row), w in zip(top_leading.iterrows(), weights):
        adj_w = w * exposure_scale
        lines.append(
            f"• **#{int(row['global_rank'])} {row['ticker']} ({row['name']})**：倉位 `{adj_w:.1f}%`｜"
            f"現價 `${row['close']:.2f}`｜{ATR_MULTIPLIER:g}xATR止損 **`${row['atr_stop_price']:.2f}`** (`-{row['atr_stop_pct']:.1f}%`)"
        )

    header = f"**⚖️ 全市場 Top 5 Leading 等風險配倉與 {ATR_PERIOD}D ATR 止損**"
    if not is_bull_market:
        header += "\n*⚠️ 空頭警戒模式：總股票倉位自動減半至 50%，其餘建議持有現金／美債*"

    return header + "\n" + "\n".join(lines)


def send_single_embed(webhook_url, embed, label):
    """獨立發送單張 Embed，徹底避開 6,000 字元總上限並處理 429 限速"""
    payload = {"embeds": [embed]}
    for attempt in range(2):
        response = requests.post(webhook_url, json=payload, timeout=30)
        if response.status_code in (200, 204):
            print(f"✅ 成功發送：{label}")
            return
        elif response.status_code == 429 and attempt == 0:
            retry_after = float(response.json().get("retry_after", 2.0))
            print(f"⏳ 觸發 Discord 限速，等待 {retry_after:.1f} 秒後重試：{label}")
            time.sleep(retry_after + 0.5)
        else:
            raise RuntimeError(f"❌ [{label}] 發送失敗，狀態碼：{response.status_code}, 回應：{response.text}")


def send_to_discord(df, prev_history, market_regime, trade_date, is_bull_market):
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL', '').strip()
    if not webhook_url:
        raise RuntimeError("❌ 錯誤：未設定 DISCORD_WEBHOOK_URL 環境變數（或為空值）")

    df['global_rank'] = df.reset_index().index + 1

    df_core = df[df['group'] == 'core'].copy()
    df_sub = df[df['group'] == 'sub_thematic'].copy()
    df_macro = df[df['group'] == 'macro_commodity'].copy()

    table_core, rot_core = generate_table_string(df_core, prev_history, trade_date)
    table_sub, rot_sub = generate_table_string(df_sub, prev_history, trade_date)
    table_macro, rot_macro = generate_table_string(df_macro, prev_history, trade_date)

    # 彙整全市場轉倉快訊：優先顯示轉入 Leading / Improving 的訊號，最多顯示 25 條防爆字數
    all_rotations_sorted = sorted(rot_core + rot_sub + rot_macro, key=lambda x: (x[0], x[1]))
    rotation_lines = [item[2] for item in all_rotations_sorted]
    if rotation_lines:
        shown_lines = rotation_lines[:25]
        extra_note = f"\n*...另外尚有 {len(rotation_lines) - 25} 檔跨象限變動*" if len(rotation_lines) > 25 else ""
        rotation_section = "**🔄 全市場 RRG 跨象限轉倉快訊（轉強優先）**\n" + "\n".join(shown_lines) + extra_note
    else:
        rotation_section = "**🔄 全市場 RRG 跨象限轉倉快訊**\n*今日暫無板塊發生跨象限轉倉.*"

    quad_counts = df['quadrant'].value_counts()
    market_breadth_summary = (
        f"**📈 全市場板塊象限分佈統計 (共 {len(df)} 檔)**\n"
        f"🟢 Leading: `{quad_counts.get('Leading', 0)}` | 🔵 Improving: `{quad_counts.get('Improving', 0)}` | "
        f"🟡 Weakening: `{quad_counts.get('Weakening', 0)}` | 🔴 Lagging: `{quad_counts.get('Lagging', 0)}`"
    )

    vol_sizing_section = build_vol_sizing_summary(df, is_bull_market)
    current_time_iso = datetime.now(timezone.utc).isoformat()

    embeds = [
        (
            "Embed 1/4 核心板塊",
            {
                "title": f"📊 [1/3] 傳統核心板塊與權重行業 ({len(df_core)} 檔)",
                "description": f"**{market_regime}**\n\n```text\n{table_core}\n```",
                "color": 3066993  # 翡翠綠
            }
        ),
        (
            "Embed 2/4 前沿科技賽道",
            {
                "title": f"🚀 [2/3] 前沿科技、AI 基建與細分賽道 ({len(df_sub)} 檔)",
                "description": f"```text\n{table_sub}\n```",
                "color": 3447003  # 極光藍
            }
        ),
        (
            "Embed 3/4 大宗與宏觀輪動",
            {
                "title": f"🌍 [3/3] 大宗資源、週期與跨資產宏觀 ({len(df_macro)} 檔)",
                "description": f"```text\n{table_macro}\n```",
                "color": 15844367  # 琥珀金
            }
        ),
        (
            "Embed 4/4 全市場總覽看板",
            {
                "title": f"🧭 全市場 {len(df)} 檔輪動總結｜轉倉快訊與等風險配倉",
                "description": (
                    f"{market_breadth_summary}\n\n"
                    f"{rotation_section}\n\n"
                    f"{vol_sizing_section}"
                ),
                "color": 5793266,  # Discord 藍紫
                "footer": {
                    "text": f"美股數據日：{trade_date} • 風險調整動能 • {ATR_PERIOD}D ATR x{ATR_MULTIPLIER:g} 止損"
                },
                "timestamp": current_time_iso
            }
        )
    ]

    for idx, (label, embed) in enumerate(embeds):
        send_single_embed(webhook_url, embed, label)
        if idx < len(embeds) - 1:
            time.sleep(0.8)  # 間隔 0.8 秒避免觸發 Discord Webhook 限速


# ==========================================
# 5. 主程式入口
# ==========================================
if __name__ == "__main__":
    prev_history = load_previous_history()
    df_result, market_regime, trade_date, is_bull_market = calculate_rrg_and_rs()
    send_to_discord(df_result, prev_history, market_regime, trade_date, is_bull_market)
    save_current_history(df_result, trade_date)
