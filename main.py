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
EQUAL_WEIGHT_BENCHMARK = 'RSP'
VIX_TICKER = '^VIX'
VIX3M_TICKER = '^VIX3M'
DXY_TICKER = 'DX-Y.NYB'
TNX_TICKER = '^TNX'
HISTORY_FILE = 'history.json'

tickers = list(SECTORS.keys()) + [
    BENCHMARK,
    EQUAL_WEIGHT_BENCHMARK,
    VIX_TICKER,
    VIX3M_TICKER,
    DXY_TICKER,
    TNX_TICKER
]

# ==========================================
# 止損、溫度計排除、起漲突破提示與標準化 RRG 參數
# ==========================================
ATR_PERIOD = 14                  # ATR 計算週期（交易日）
ATR_MULTIPLIER = 2.0             # 動態追蹤止損倍數

RS_STREAK_DAYS = 3               # 觸發提示所需嘅連續 RS 上升交易日數
RS_STREAK_MIN_RATING = 60        # 起漲突破門檻（由低於 60 升穿至 60 或以上）
RS_BREAKOUT_ONCE_ONLY = True     # True = 單次邊緣觸發（每段突破只提示 1 次，防第 4、5 日洗版）
RS_BREAKOUT_INCLUDE_SPY = False  # 如想同時捕捉「剛由跑輸轉為跑贏 SPY」嘅起漲點，可改 True

# 僅作市場溫度計、排除出 Top 5 股票強勢榜嘅宏觀／非股票資產
THERMOMETER_ONLY_TICKERS = {'TLT', 'HYG', 'GLD', 'SLV', 'IBIT'}

RRG_RATIO_FAST = 10              # RS-Ratio 短週期 EMA
RRG_RATIO_SLOW = 40              # RS-Ratio 長週期 EMA
RRG_MOM_FAST = 5                 # RS-Momentum 短週期 EMA
RRG_MOM_SLOW = 15                # RS-Momentum 長週期 EMA
RRG_STD_WINDOW = 60              # 100 軸錨定標準化滾動波動率視窗（交易日）
RRG_DEADBAND_SIGMA = 0.35        # 100 軸防假轉倉遲滯緩衝區（單位：σ，建議 0.30 至 0.50）

# 象限中文顯示對照（history.json 底層維持英文不變，無縫相容）
QUADRANT_ZH = {
    "Leading": "領先",
    "Improving": "改善",
    "Weakening": "轉弱",
    "Lagging": "落後"
}


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
# 3. EMA 標準化 RRG、起漲突破邊緣偵測、宏觀儀表板與 ATR 計算
# ==========================================
def get_group_name(ticker):
    if ticker in CORE_SECTORS:
        return 'core'
    elif ticker in SUB_THEMATIC_SECTORS:
        return 'sub_thematic'
    return 'macro_commodity'


def get_quadrant_emoji(quad):
    mapping = {
        "Leading": "🟢",
        "Improving": "🔵",
        "Weakening": "🟡",
        "Lagging": "🔴"
    }
    return mapping.get(quad, "⚪")


def calc_ema_standardized_rrg(price_series, bench_series):
    """
    計算 EMA 驅動 + 100 軸錨定波幅標準化嘅 RS-Ratio 與 RS-Momentum
    回傳：(ratio_z, mom_z)，單位為相對 100 基準軸嘅標準差倍數 (σ)
    """
    aligned = pd.concat(
        [price_series.rename('asset'), bench_series.rename('bench')],
        axis=1,
        join='inner'
    ).dropna()
    aligned = aligned[(aligned['asset'] > 0) & (aligned['bench'] > 0)]

    if len(aligned) < 20:
        return 0.0, 0.0

    rel = aligned['asset'] / aligned['bench']
    ema_short = rel.ewm(span=RRG_RATIO_FAST, adjust=False).mean()
    ema_long = rel.ewm(span=RRG_RATIO_SLOW, adjust=False).mean()
    raw_ratio = 100.0 * (ema_short / ema_long)

    mom_short = raw_ratio.ewm(span=RRG_MOM_FAST, adjust=False).mean()
    mom_long = raw_ratio.ewm(span=RRG_MOM_SLOW, adjust=False).mean()
    raw_mom = 100.0 * (mom_short / mom_long)

    ratio_std = raw_ratio.rolling(window=RRG_STD_WINDOW, min_periods=20).std().fillna(1.0).clip(lower=0.30)
    mom_std = raw_mom.rolling(window=RRG_STD_WINDOW, min_periods=20).std().fillna(0.50).clip(lower=0.15)

    ratio_z_series = (raw_ratio - 100.0) / ratio_std
    mom_z_series = (raw_mom - 100.0) / mom_std

    latest_ratio_z = float(ratio_z_series.iloc[-1]) if pd.notna(ratio_z_series.iloc[-1]) else 0.0
    latest_mom_z = float(mom_z_series.iloc[-1]) if pd.notna(mom_z_series.iloc[-1]) else 0.0
    return latest_ratio_z, latest_mom_z


def resolve_quadrant_with_deadband(ratio_z, mom_z, prev_quad=None, deadband=RRG_DEADBAND_SIGMA):
    """雙軸遲滯防假轉倉緩衝區判定：±deadband 區間內延用上一交易日方向"""
    raw_ratio_pos = ratio_z >= 0.0
    raw_mom_pos = mom_z >= 0.0

    if prev_quad in ("Leading", "Improving", "Weakening", "Lagging"):
        prev_ratio_pos = prev_quad in ("Leading", "Weakening")
        prev_mom_pos = prev_quad in ("Leading", "Improving")

        if ratio_z >= deadband:
            final_ratio_pos = True
        elif ratio_z <= -deadband:
            final_ratio_pos = False
        else:
            final_ratio_pos = prev_ratio_pos

        if mom_z >= deadband:
            final_mom_pos = True
        elif mom_z <= -deadband:
            final_mom_pos = False
        else:
            final_mom_pos = prev_mom_pos
    else:
        final_ratio_pos = raw_ratio_pos
        final_mom_pos = raw_mom_pos

    def _map_quad(r_pos, m_pos):
        if r_pos and m_pos:
            return "Leading", "🟢"
        elif not r_pos and m_pos:
            return "Improving", "🔵"
        elif r_pos and not m_pos:
            return "Weakening", "🟡"
        return "Lagging", "🔴"

    raw_quad, _ = _map_quad(raw_ratio_pos, raw_mom_pos)
    final_quad, final_emoji = _map_quad(final_ratio_pos, final_mom_pos)
    held_by_deadband = (raw_quad != final_quad)
    return final_quad, final_emoji, held_by_deadband


def calc_virtual_rs_rating(score, ref_scores):
    """將基準標的 (SPY 或 RSP) 嘅原始 RS 分數映射至 60 檔板塊嘅 1–99 百分位評分"""
    n = len(ref_scores)
    if n <= 1:
        return 99 if score >= 0 else 1
    below = int((ref_scores < score).sum())
    equal = int((ref_scores == score).sum())
    virtual_rank = below + (equal * 0.5) + 0.5
    rating = int(round(1 + (virtual_rank - 1) / (n - 1) * 98))
    return max(1, min(99, rating))


def _is_breakout_at(r_vals, s_vals, idx):
    """檢查第 idx 日是否符合：連續 N 日 RS 上升，且 N 日內剛由低於門檻突破起漲"""
    if idx < RS_STREAK_DAYS:
        return False
    for j in range(idx, idx - RS_STREAK_DAYS, -1):
        if r_vals[j] <= r_vals[j - 1]:
            return False

    start_idx = idx - RS_STREAK_DAYS
    crossed_min = (r_vals[idx] >= RS_STREAK_MIN_RATING) and (r_vals[start_idx] < RS_STREAK_MIN_RATING)
    if not RS_BREAKOUT_INCLUDE_SPY:
        return crossed_min

    crossed_spy = (s_vals[idx] > 0.0) and (s_vals[start_idx] <= 0.0)
    return crossed_min or crossed_spy


def detect_rs_breakout(rating_vals, score_vals):
    """單次邊緣觸發：今日符合起漲突破，且昨日尚未觸發（防連升第 4、5 日重複洗版）"""
    n = len(rating_vals)
    if n <= RS_STREAK_DAYS:
        return False
    today_ok = _is_breakout_at(rating_vals, score_vals, n - 1)
    if not today_ok:
        return False
    if RS_BREAKOUT_ONCE_ONLY and n > RS_STREAK_DAYS + 1:
        yesterday_ok = _is_breakout_at(rating_vals, score_vals, n - 2)
        return not yesterday_ok
    return True


def build_vix_metrics(raw_vix_series, raw_vix3m_series=None):
    """獨立計算 VIX 讀數、20D 均線，以及 VIX / VIX3M 期限結構（手機雙層精簡版）"""
    if raw_vix_series is None or raw_vix_series.empty:
        return "⚪ **VIX** 暫未能取得數據\n⚪ **VIX/VIX3M** 暫未能取得數據", None

    vix_latest = float(raw_vix_series.iloc[-1])
    if len(raw_vix_series) >= 2:
        vix_prev = float(raw_vix_series.iloc[-2])
        vix_chg_pct = ((vix_latest - vix_prev) / vix_prev * 100.0) if vix_prev > 0 else 0.0
    else:
        vix_chg_pct = 0.0

    vix_sma20 = (
        float(raw_vix_series.rolling(window=20).mean().iloc[-1])
        if len(raw_vix_series) >= 20
        else vix_latest
    )

    if vix_latest < 15:
        vix_emoji, vix_status = "🟢", "低波平靜"
    elif vix_latest < 20:
        vix_emoji, vix_status = "🟡", "正常波幅"
    elif vix_latest < 30:
        vix_emoji, vix_status = "🟠", "警戒升溫"
    else:
        vix_emoji, vix_status = "🔴", "極度恐慌⚠️"

    vix_trend = "高於均線" if vix_latest > vix_sma20 else "低於均線"
    vix_block = (
        f"{vix_emoji} **VIX** `{vix_latest:.2f}` (`{vix_chg_pct:+.1f}%`) · {vix_status}\n"
        f"-#  20D均 {vix_sma20:.2f} ({vix_trend})"
    )

    term_ratio = None
    vix3m_latest = None
    term_emoji = "⚪"
    term_title = "數據缺失"
    term_note = "暫未能取得 VIX3M"
    term_block = "⚪ **VIX/VIX3M** 暫未能取得數據"

    if raw_vix3m_series is not None and not raw_vix3m_series.empty:
        aligned = pd.concat(
            [raw_vix_series.rename('vix'), raw_vix3m_series.rename('vix3m')],
            axis=1,
            join='inner'
        ).dropna()
        aligned = aligned[aligned['vix3m'] > 0]

        if not aligned.empty:
            ratio_series = aligned['vix'] / aligned['vix3m']
            term_ratio = float(ratio_series.iloc[-1])
            ratio_prev = float(ratio_series.iloc[-2]) if len(ratio_series) >= 2 else term_ratio
            ratio_5d_max = float(ratio_series.tail(5).max())
            vix3m_latest = float(aligned['vix3m'].iloc[-1])

            if term_ratio >= 1.15:
                term_emoji = "🔴"
                if term_ratio < ratio_prev:
                    term_title, term_note = "極端倒掛回落⚡", "恐慌高潮／留意見底"
                else:
                    term_title, term_note = "極端倒掛⚠️", "投降拋售中，勿急接刀"
            elif term_ratio >= 1.00:
                term_emoji = "🟠"
                if ratio_5d_max >= 1.10 and term_ratio < ratio_prev:
                    term_title, term_note = "倒掛收斂回落⚡", "短線築底觀察"
                else:
                    term_title, term_note = "期限倒掛", "短線恐慌升溫"
            elif term_ratio >= 0.92:
                if ratio_5d_max >= 1.02 and term_ratio < ratio_prev:
                    term_emoji = "🟢"
                    term_title, term_note = "重返正價差⚡", "短線見底確認"
                else:
                    term_emoji = "🟡"
                    term_title, term_note = "結構趨平", "避險需求上升"
            else:
                term_emoji = "🟢"
                if ratio_5d_max >= 1.02:
                    term_title, term_note = "恐慌出清⚡", "重返正價差／見底確認"
                else:
                    term_title, term_note = "正價差 Contango", "結構健康"

            term_block = (
                f"{term_emoji} **VIX/VIX3M** `{term_ratio:.2f}` · {term_title}\n"
                f"-#  VIX3M {vix3m_latest:.2f} · {term_note}"
            )

    vix_info = {
        "latest": vix_latest,
        "chg_pct": vix_chg_pct,
        "sma20": vix_sma20,
        "emoji": vix_emoji,
        "status": vix_status,
        "vix3m": vix3m_latest,
        "term_ratio": term_ratio,
        "term_emoji": term_emoji,
        "term_title": term_title,
        "term_note": term_note
    }
    return f"{vix_block}\n{term_block}", vix_info


def build_dxy_tnx_metrics(raw_dxy_series, raw_tnx_series):
    """獨立計算美元指數 (DXY) 與 10 年期美債息 (US10Y)，以手機雙層格式輸出"""
    if raw_dxy_series is None or raw_dxy_series.empty:
        dxy_block = "⚪ **DXY 美元** 暫未能取得數據"
    else:
        dxy_latest = float(raw_dxy_series.iloc[-1])
        if len(raw_dxy_series) >= 2:
            dxy_prev = float(raw_dxy_series.iloc[-2])
            dxy_chg_pct = ((dxy_latest - dxy_prev) / dxy_prev * 100.0) if dxy_prev > 0 else 0.0
        else:
            dxy_chg_pct = 0.0

        dxy_sma20 = (
            float(raw_dxy_series.rolling(window=20
