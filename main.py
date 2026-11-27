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
# 風控、溫度計排除、起漲突破提示與標準化 RRG 參數
# ==========================================
TARGET_POS_VOL = 3.0             # 單一倉位目標年化波動貢獻 (%)
MAX_POS_WEIGHT = 25.0            # 單一板塊最大倉位上限 (%)
ATR_PERIOD = 14                  # ATR 計算週期（交易日）
ATR_MULTIPLIER = 2.0             # 動態追蹤止損倍數

RS_STREAK_DAYS = 3               # 觸發提示所需嘅連續 RS 上升交易日數
RS_STREAK_MIN_RATING = 60        # 起漲突破門檻（由低於 60 升穿至 60 或以上）
RS_BREAKOUT_ONCE_ONLY = True     # True = 單次邊緣觸發（每段突破只提示 1 次，防第 4、5 日洗版）
RS_BREAKOUT_INCLUDE_SPY = False  # 如想同時捕捉「剛由跑輸轉為跑贏 SPY」嘅起漲點，可改 True

# 僅作市場溫度計、排除出 Top 5 股票板塊配倉嘅宏觀／非股票資產
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
            float(raw_dxy_series.rolling(window=20).mean().iloc[-1])
            if len(raw_dxy_series) >= 20
            else dxy_latest
        )
        dxy_sma50 = (
            float(raw_dxy_series.rolling(window=50).mean().iloc[-1])
            if len(raw_dxy_series) >= 50
            else dxy_sma20
        )

        if dxy_latest > dxy_sma20 and dxy_latest > dxy_sma50:
            dxy_emoji, dxy_status, dxy_pos = "🟠", "偏強上行", "高於雙均線"
        elif dxy_latest < dxy_sma20 and dxy_latest < dxy_sma50:
            dxy_emoji, dxy_status, dxy_pos = "🟢", "偏軟回落", "低於雙均線"
        else:
            dxy_emoji, dxy_status, dxy_pos = "🟡", "區間震盪", "均線之間"

        dxy_block = (
            f"{dxy_emoji} **DXY 美元** `{dxy_latest:.2f}` (`{dxy_chg_pct:+.2f}%`) · {dxy_status}\n"
            f"-#  20D均 {dxy_sma20:.2f} · 50D均 {dxy_sma50:.2f} ({dxy_pos})"
        )

    if raw_tnx_series is None or raw_tnx_series.empty:
        tnx_block = "⚪ **US10Y 債息** 暫未能取得數據"
    else:
        tnx_s = raw_tnx_series.copy()
        if float(tnx_s.iloc[-1]) > 20.0:
            tnx_s = tnx_s / 10.0

        tnx_latest = float(tnx_s.iloc[-1])
        tnx_chg_bp = (tnx_latest - float(tnx_s.iloc[-2])) * 100.0 if len(tnx_s) >= 2 else 0.0
        tnx_5d_bp = (tnx_latest - float(tnx_s.iloc[-6])) * 100.0 if len(tnx_s) >= 6 else 0.0
        tnx_sma20 = (
            float(tnx_s.rolling(window=20).mean().iloc[-1])
            if len(tnx_s) >= 20
            else tnx_latest
        )

        if tnx_latest > tnx_sma20 and tnx_5d_bp >= 15.0:
            tnx_emoji, tnx_status = "🔴", "急升壓估值⚠️"
        elif tnx_latest > tnx_sma20:
            tnx_emoji, tnx_status = "🟠", "偏強上行"
        elif tnx_latest < tnx_sma20 and tnx_5d_bp <= -15.0:
            tnx_emoji, tnx_status = "🟢", "顯著回落（估值鬆綁）"
        elif tnx_latest < tnx_sma20:
            tnx_emoji, tnx_status = "🟢", "偏軟回落"
        else:
            tnx_emoji, tnx_status = "🟡", "持平震盪"

        tnx_trend = "高於20D均" if tnx_latest > tnx_sma20 else "低於20D均"
        tnx_block = (
            f"{tnx_emoji} **US10Y 債息** `{tnx_latest:.2f}%` (`{tnx_chg_bp:+.1f}bp`) · {tnx_status}\n"
            f"-#  20D均 {tnx_sma20:.2f}% · 5D {tnx_5d_bp:+.0f}bp ({tnx_trend})"
        )

    return f"{dxy_block}\n{tnx_block}"


def calculate_rrg_and_rs(prev_history):
    print(f"正在下載全市場 {len(SECTORS)} 檔 ETF 及 SPY / RSP / VIX / VIX3M / DXY / US10Y 數據...")

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

    # 在對齊與 ffill 之前，獨立抽出非美股時段宏觀序列，確保單日變動唔被假期補值稀釋
    raw_vix_series = data[VIX_TICKER].dropna() if VIX_TICKER in data.columns else pd.Series(dtype=float)
    raw_vix3m_series = data[VIX3M_TICKER].dropna() if VIX3M_TICKER in data.columns else pd.Series(dtype=float)
    raw_dxy_series = data[DXY_TICKER].dropna() if DXY_TICKER in data.columns else pd.Series(dtype=float)
    raw_tnx_series = data[TNX_TICKER].dropna() if TNX_TICKER in data.columns else pd.Series(dtype=float)

    if BENCHMARK not in data.columns:
        raise ValueError(f"❌ 錯誤：未能取得基準 {BENCHMARK} 數據。")

    # 強制鎖死對齊 SPY 嘅美股交易日期，防止外匯非美股交易日多出一行導致 ETF 單日變動歸零
    spy_valid_dates = data[BENCHMARK].dropna().index
    data = data.reindex(spy_valid_dates).dropna(axis=1, how='all').ffill()
    high_df = high_df.reindex(index=spy_valid_dates, columns=data.columns).ffill()
    low_df = low_df.reindex(index=spy_valid_dates, columns=data.columns).ffill()

    if len(data) < 200:
        raise ValueError(f"❌ 錯誤：下載的 {BENCHMARK} 數據不足 200 筆（目前只有 {len(data)} 筆），無法計算 SMA200。")

    # 1. 美股交易日期及 SPY 大市環境 (SMA200)
    spy_series = data[BENCHMARK].dropna()
    trade_date = spy_series.index[-1].strftime('%Y-%m-%d')
    spy_sma200 = spy_series.rolling(window=200).mean().iloc[-1]
    spy_latest = spy_series.iloc[-1]
    is_bull_market = bool(spy_latest >= spy_sma200)

    # 2. 向量化計算多週期相對回報率、20 日波動率及近 15 日 IBD 1–99 RS 歷史軌跡
    daily_returns = data.pct_change(1, fill_method=None)
    ret_5d_df = data.pct_change(5, fill_method=None)
    ret_2w_df = data.pct_change(10, fill_method=None)
    ret_1m_df = data.pct_change(21, fill_method=None)
    vol_20d_daily_df = (daily_returns.rolling(window=20).std() * 100.0).fillna(1.0).clip(lower=0.2)

    sector_cols = [t for t in SECTORS.keys() if t in data.columns]
    e1d_df = daily_returns[sector_cols].sub(daily_returns[BENCHMARK], axis=0).fillna(0.0) * 100.0
    e5d_df = ret_5d_df[sector_cols].sub(ret_5d_df[BENCHMARK], axis=0).fillna(0.0) * 100.0
    e2w_df = ret_2w_df[sector_cols].sub(ret_2w_df[BENCHMARK], axis=0).fillna(0.0) * 100.0
    e1m_df = ret_1m_df[sector_cols].sub(ret_1m_df[BENCHMARK], axis=0).fillna(0.0) * 100.0

    weighted_df = (e1d_df * 0.10) + (e5d_df * 0.30) + (e2w_df * 0.30) + (e1m_df * 0.30)
    rs_score_hist_df = weighted_df / vol_20d_daily_df[sector_cols]

    # 逐日橫向轉換為 1–99 IBD RS 評分，即時追蹤連續上升日數與單次起漲突破
    recent_scores = rs_score_hist_df.tail(15)
    n_cols = len(sector_cols)
    if n_cols > 1:
        asc_rank_hist = recent_scores.rank(axis=1, method='average', ascending=True)
        rs_rating_hist_df = (1 + (asc_rank_hist - 1) / (n_cols - 1) * 98).round().astype(int).clip(1, 99)
    else:
        rs_rating_hist_df = pd.DataFrame(99, index=recent_scores.index, columns=sector_cols)

    ret_1d = daily_returns.iloc[-1]
    ret_5d = ret_5d_df.iloc[-1]
    ret_2w = ret_2w_df.iloc[-1]
    ret_1m = ret_1m_df.iloc[-1]
    vol_20d_daily = vol_20d_daily_df.iloc[-1]

    spy_1d = ret_1d[BENCHMARK]
    spy_5d = ret_5d[BENCHMARK]
    spy_2w = ret_2w[BENCHMARK]
    spy_1m = ret_1m[BENCHMARK]

    results = []
    for t, name in SECTORS.items():
        if t in sector_cols:
            safe_vol = float(vol_20d_daily[t])
            ann_vol_20d = safe_vol * np.sqrt(252)

            rs_score = float(rs_score_hist_df[t].iloc[-1])
            rs_rating = int(rs_rating_hist_df[t].iloc[-1])
            pos_size = min((TARGET_POS_VOL / ann_vol_20d) * 100.0, MAX_POS_WEIGHT)

            # 計算連續 RS 評分上升日數、近 4 日軌跡及「剛突破起漲點」單次邊緣觸發
            rating_vals = rs_rating_hist_df[t].values
            score_vals = recent_scores[t].values
            is_rs_breakout = detect_rs_breakout(rating_vals, score_vals)

            rs_up_streak = 0
            for k in range(len(rating_vals) - 1, 0, -1):
                if rating_vals[k] > rating_vals[k - 1]:
                    rs_up_streak += 1
                else:
                    break

            streak_start_idx = max(0, len(rating_vals) - 1 - rs_up_streak)
            rs_streak_delta = int(rating_vals[-1] - rating_vals[streak_start_idx]) if rs_up_streak > 0 else 0
            rs_trail = "➔".join(f"{int(v):02d}" for v in rating_vals[-4:]) if len(rating_vals) >= 4 else ""

            # 14 日 Wilder ATR 動態止損位
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

            # EMA + 100 軸錨定波幅標準化 RRG + 上一交易日遲滯防假轉倉緩衝區
            ratio_z, mom_z = calc_ema_standardized_rrg(close_s, spy_series)
            item_hist = prev_history.get(t, {})
            past_records = [r for r in item_hist.get('records', []) if r.get('date') != trade_date]
            prev_quad = past_records[-1].get('quadrant') if past_records else None

            quadrant, quad_emoji, held_by_deadband = resolve_quadrant_with_deadband(
                ratio_z, mom_z, prev_quad=prev_quad, deadband=RRG_DEADBAND_SIGMA
            )

            results.append({
                'ticker': t,
                'name': name,
                'group': get_group_name(t),
                'close': latest_close,
                'atr_stop_price': atr_stop_price,
                'atr_stop_pct': atr_stop_pct,
                'ann_vol_20d': ann_vol_20d,
                'pos_size': pos_size,
                'rs_score': rs_score,
                'rs_rating': rs_rating,
                'rs_up_streak': rs_up_streak,
                'rs_streak_delta': rs_streak_delta,
                'rs_trail': rs_trail,
                'is_rs_breakout': is_rs_breakout,
                'ratio_z': ratio_z,
                'mom_z': mom_z,
                'quadrant': quadrant,
                'quad_emoji': quad_emoji,
                'held_by_deadband': held_by_deadband
            })

    df_rs = pd.DataFrame(results)
    df_rs = df_rs.sort_values(by='rs_score', ascending=False).reset_index(drop=True)

    # 3. 組合手機版頂部 SPY / RSP / VIX / VIX3M / DXY / US10Y 宏觀儀表板
    n = len(df_rs)
    beat_count = int((df_rs['rs_score'] > 0).sum())
    if beat_count == n:
        spy_rs_display = "RS <01"
    elif beat_count == 0:
        spy_rs_display = "RS >99"
    else:
        spy_rs_display = f"RS {calc_virtual_rs_rating(0.0, df_rs['rs_score']):02d}"

    if is_bull_market:
        spy_line = f"🟢 **SPY** `{spy_rs_display}` · 多頭牛市 (企穩 SMA200)"
    else:
        spy_line = f"🔴 **SPY** `{spy_rs_display}` · 空頭警戒 (跌破 SMA200)"

    rsp_info = None
    if EQUAL_WEIGHT_BENCHMARK in data.columns:
        rsp_series = data[EQUAL_WEIGHT_BENCHMARK].dropna()
        rsp_latest = float(rsp_series.iloc[-1])
        if len(rsp_series) >= 200:
            rsp_sma200 = float(rsp_series.rolling(window=200).mean().iloc[-1])
            rsp_above_sma200 = rsp_latest >= rsp_sma200
            rsp_sma_emoji = "🟢" if rsp_above_sma200 else "🔴"
            rsp_sma_text = "企穩 SMA200" if rsp_above_sma200 else "跌破 SMA200"
        else:
            rsp_sma_emoji = "⚪"
            rsp_sma_text = "SMA200不足"

        rsp_e1d = float((ret_1d[EQUAL_WEIGHT_BENCHMARK] - spy_1d) * 100) if pd.notna(ret_1d[EQUAL_WEIGHT_BENCHMARK]) else 0.0
        rsp_e5d = float((ret_5d[EQUAL_WEIGHT_BENCHMARK] - spy_5d) * 100) if pd.notna(ret_5d[EQUAL_WEIGHT_BENCHMARK]) else 0.0
        rsp_e2w = float((ret_2w[EQUAL_WEIGHT_BENCHMARK] - spy_2w) * 100) if pd.notna(ret_2w[EQUAL_WEIGHT_BENCHMARK]) else 0.0
        rsp_e1m = float((ret_1m[EQUAL_WEIGHT_BENCHMARK] - spy_1m) * 100) if pd.notna(ret_1m[EQUAL_WEIGHT_BENCHMARK]) else 0.0

        rsp_weighted = (rsp_e1d * 0.10) + (rsp_e5d * 0.30) + (rsp_e2w * 0.30) + (rsp_e1m * 0.30)
        rsp_raw_vol = (
            float(vol_20d_daily[EQUAL_WEIGHT_BENCHMARK])
            if EQUAL_WEIGHT_BENCHMARK in vol_20d_daily.index and pd.notna(vol_20d_daily[EQUAL_WEIGHT_BENCHMARK])
            else 1.0
        )
        rsp_rs_score = rsp_weighted / max(rsp_raw_vol, 0.2)
        rsp_rs_rating = calc_virtual_rs_rating(rsp_rs_score, df_rs['rs_score'])

        rsp_ratio_z, rsp_mom_z = calc_ema_standardized_rrg(rsp_series, spy_series)
        rsp_quad, rsp_quad_emoji, _ = resolve_quadrant_with_deadband(
            rsp_ratio_z, rsp_mom_z, prev_quad=None, deadband=0.0
        )
        rsp_quad_zh = QUADRANT_ZH.get(rsp_quad, rsp_quad)

        if rsp_rs_score > 0:
            rsp_vs_spy = "跑贏"
        elif rsp_rs_score < 0:
            rsp_vs_spy = "跑輸"
        else:
            rsp_vs_spy = "持平"

        rsp_line = (
            f"{rsp_sma_emoji} **RSP** `RS {rsp_rs_rating:02d}` · {rsp_sma_text}\n"
            f"-#  {rsp_quad_emoji}{rsp_quad_zh}({rsp_ratio_z:+.1f}σ) · {rsp_vs_spy} SPY (等權平均參考)"
        )
        rsp_info = {
            "rs_rating": rsp_rs_rating,
            "rs_score": rsp_rs_score,
            "quadrant": rsp_quad,
            "ratio_z": rsp_ratio_z,
            "vs_spy": rsp_vs_spy
        }
    else:
        rsp_line = "⚪ **RSP** 暫未能取得數據"

    vix_block, vix_info = build_vix_metrics(raw_vix_series, raw_vix3m_series)
    dxy_tnx_block = build_dxy_tnx_metrics(raw_dxy_series, raw_tnx_series)

    market_regime = (
        f"### 📡 大市環境｜{trade_date}\n"
        f"{spy_line}\n"
        f"{rsp_line}\n"
        f"{vix_block}\n"
        f"{dxy_tnx_block}"
    )
    return df_rs, market_regime, trade_date, is_bull_market, rsp_info, vix_info


# ==========================================
# 4. 手機兩行細字清晰卡片排版與 Discord 四張獨立 Embed 發送（總覽置頂）
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


def generate_mobile_list(sub_df, prev_history, trade_date):
    """生成手機直向專用三層清晰卡片清單（含單次起漲突破 🔥 特別標記）"""
    lines = []
    rotations = []

    for _, row in sub_df.iterrows():
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
        quad_zh = QUADRANT_ZH.get(quad, quad)
        ratio_z = float(row['ratio_z'])
        mom_z = float(row['mom_z'])
        held_by_deadband = bool(row['held_by_deadband'])
        rs_up_streak = int(row.get('rs_up_streak', 0))

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
                quality_str = "剛轉強"
            else:
                quality_str = f"強{leading_days}天"
        elif rank_std > 3.5:
            quality_str = "波動⚠️"
        elif rank < avg_rank_5d - 1:
            quality_str = "穩升📈"
        elif rank > avg_rank_5d + 1:
            quality_str = "回落📉"
        else:
            quality_str = "橫盤"

        has_rotated = False
        if prev_quad and prev_quad != quad:
            has_rotated = True
            prev_emoji = get_quadrant_emoji(prev_quad)
            prev_quad_zh = QUADRANT_ZH.get(prev_quad, prev_quad)
            priority = 0 if quad == "Leading" else (1 if quad == "Improving" else 2)
            rotations.append((
                priority,
                rank,
                f"▸ **#{rank:02d} {ticker}** {name} `RS {int(row['rs_rating']):02d}`\n"
                f"-#  {prev_emoji}{prev_quad_zh} ➔ {emoji}{quad_zh} (趨勢 {ratio_z:+.1f}σ / 動能 {mom_z:+.1f}σ)"
            ))

        if has_rotated:
            quad_tag = f"{quad_zh}⭐"
        elif held_by_deadband:
            quad_tag = f"{quad_zh}🛡️"
        else:
            quad_tag = quad_zh

        rs_badge = f"RS {int(row['rs_rating']):02d}"
        is_streak = bool(row.get('is_rs_breakout', False))
        streak_icon = " 🔥" if is_streak else ""
        streak_sub = f" · 🔥剛穿{RS_STREAK_MIN_RATING}(連升{rs_up_streak}日)" if is_streak else ""

        if ticker in THERMOMETER_ONLY_TICKERS:
            pos_label = "🌡️溫度計" if is_streak else "🌡️宏觀溫度計（不入配倉）"
        else:
            pos_label = f"建議單倉 {row['pos_size']:.0f}%"

        line_main = f"{emoji} **#{rank:02d} {ticker}** {name} `{rs_badge}` {change_str}{streak_icon}"
        line_sub_1 = f"-#  {quad_tag}({ratio_z:+.1f}σ) · {quality_str} · 5日均名 {avg_rank_5d:.1f}"
        line_sub_2 = f"-#  年波 {row['ann_vol_20d']:.0f}% · {pos_label}{streak_sub}"
        lines.append(f"{line_main}\n{line_sub_1}\n{line_sub_2}")

    return "\n".join(lines), rotations


def build_rs_streak_summary(df):
    """彙整全市場剛由低於門檻突破起漲、且連續 RS 上升嘅單次特別提示"""
    spy_suffix = "／轉贏SPY" if RS_BREAKOUT_INCLUDE_SPY else ""
    header = f"### 🔥 起漲突破提示（連升{RS_STREAK_DAYS}日剛穿{RS_STREAK_MIN_RATING}{spy_suffix}）"

    if df.empty or 'is_rs_breakout' not in df.columns:
        return f"{header}\n-# 今日暫無板塊剛由低於 {RS_STREAK_MIN_RATING} 連升突破起漲"

    streak_df = df[df['is_rs_breakout']].copy()
    if streak_df.empty:
        return f"{header}\n-# 今日暫無板塊剛由低於 {RS_STREAK_MIN_RATING} 連升突破起漲"

    streak_df = streak_df.sort_values(by=['global_rank'], ascending=[True])

    lines = []
    for _, row in streak_df.head(10).iterrows():
        quad_zh = QUADRANT_ZH.get(row['quadrant'], row['quadrant'])
        thermo_tag = " · 🌡️溫度計" if row['ticker'] in THERMOMETER_ONLY_TICKERS else ""
        lines.append(
            f"🔥 **#{int(row['global_rank']):02d} {row['ticker']}** {row['name']} "
            f"`RS {int(row['rs_rating']):02d}` (連升{int(row['rs_up_streak'])}日 `+{int(row['rs_streak_delta'])}`)\n"
            f"-#  {row['quad_emoji']}{quad_zh}({float(row['ratio_z']):+.1f}σ) · 近4日 {row['rs_trail']}{thermo_tag}"
        )

    extra_note = (
        f"\n-# ...另外尚有 {len(streak_df) - 10} 檔剛起漲突破（明細見下方 🔥 標記）"
        if len(streak_df) > 10
        else ""
    )
    return header + "\n" + "\n".join(lines) + extra_note


def build_spy_benchmark_summary(df, rsp_info=None, vix_info=None):
    """手機版 SPY / RSP 基準分界線與跑贏廣度"""
    n = len(df)
    if n == 0:
        return ""

    beat_df = df[df['rs_score'] > 0]
    beat_count = len(beat_df)
    beat_pct = (beat_count / n) * 100.0

    if beat_count == n:
        spy_rs_str = "RS <01"
        threshold_str = "RS ≥ 01 (全數跑贏)"
    elif beat_count == 0:
        spy_rs_str = "RS >99"
        threshold_str = "無標的跑贏 ⚠️"
    else:
        spy_rs = calc_virtual_rs_rating(0.0, df['rs_score'])
        min_beat_rs = int(beat_df['rs_rating'].min())
        spy_rs_str = f"RS {spy_rs:02d}"
        threshold_str = f"RS ≥ {min_beat_rs:02d}"

    if beat_pct >= 60:
        breadth_tag = "廣度強勁（百花齊放）"
    elif beat_pct >= 35:
        breadth_tag = "結構分化（板塊輪動）"
    else:
        breadth_tag = "極度窄幅（權重抱團⚠️）"

    rsp_part = f" · **RSP** `RS {rsp_info['rs_rating']:02d}` ({rsp_info['vs_spy']})" if rsp_info else ""

    vix_part = ""
    if vix_info:
        term_str = (
            f" · 期限比 `{vix_info['term_ratio']:.2f}`"
            if vix_info.get('term_ratio') is not None
            else ""
        )
        vix_part = (
            f"\n▸ **VIX** {vix_info['emoji']}`{vix_info['latest']:.2f}` (`{vix_info['chg_pct']:+.1f}%`){term_str}\n"
            f"-#  {vix_info['term_emoji']} {vix_info['term_title']} · {vix_info['term_note']}"
        )

    shield_count = int(df['held_by_deadband'].sum()) if 'held_by_deadband' in df.columns else 0
    shield_note = f" · 🛡️緩衝鎖定 `{shield_count}` 檔" if shield_count > 0 else ""

    return (
        f"### 🎯 大市基準與廣度\n"
        f"▸ **SPY** `{spy_rs_str}`{rsp_part}\n"
        f"▸ **跑贏門檻** **`{threshold_str}`** · 廣度 `{beat_count}/{n} ({beat_pct:.0f}%)`\n"
        f"-#  {breadth_tag}{shield_note}"
        f"{vix_part}"
    )


def build_vol_sizing_summary(df, is_bull_market):
    """只針對股票板塊計算 Top 5 領先等風險配倉，並獨立列出領先嘅宏觀溫度計"""
    all_leading = df[df['quadrant'] == 'Leading'].copy()
    equity_leading = all_leading[~all_leading['ticker'].isin(THERMOMETER_ONLY_TICKERS)].head(5).copy()
    macro_leading = all_leading[all_leading['ticker'].isin(THERMOMETER_ONLY_TICKERS)].copy()

    if not macro_leading.empty:
        macro_tags = [
            f"{row['ticker']}({row['name']} RS {int(row['rs_rating']):02d})"
            for _, row in macro_leading.iterrows()
        ]
        macro_note = f"\n-# 🌡️ 領先宏觀溫度計：{' · '.join(macro_tags)}（已排除出配倉）"
    else:
        macro_note = "\n-# 🌡️ 宏觀溫度計：目前無債券／貴金屬／BTC 處於領先象限"

    header = f"### ⚖️ Top 5 股票板塊配倉與 {ATR_PERIOD}D ATR 止損"

    if equity_leading.empty:
        return f"{header}\n*目前暫無股票板塊處於領先象限，建議保留現金觀望。*{macro_note}"

    inv_vols = 1.0 / equity_leading['ann_vol_20d']
    weights = (inv_vols / inv_vols.sum()) * 100.0
    exposure_scale = 1.0 if is_bull_market else 0.5

    lines = []
    for (_, row), w in zip(equity_leading.iterrows(), weights):
        adj_w = w * exposure_scale
        lines.append(
            f"🟢 **#{int(row['global_rank']):02d} {row['ticker']}** {row['name']} "
            f"`RS {int(row['rs_rating']):02d}` · 倉 **`{adj_w:.1f}%`**\n"
            f"-#  現價 ${row['close']:.2f} · 止損 ${row['atr_stop_price']:.2f} (-{row['atr_stop_pct']:.1f}%) · 強{row['ratio_z']:+.1f}σ"
        )

    if not is_bull_market:
        header += "\n-# ⚠️ 空頭警戒模式：總股票倉位自動減半至 50%，其餘建議持有現金／美債"

    return header + "\n" + "\n".join(lines) + macro_note


def send_single_embed(webhook_url, embed, label):
    """獨立發送單張 Embed，避開 6,000 字元總上限並處理 429 限速"""
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


def send_to_discord(df, prev_history, market_regime, trade_date, is_bull_market, rsp_info=None, vix_info=None):
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL', '').strip()
    if not webhook_url:
        raise RuntimeError("❌ 錯誤：未設定 DISCORD_WEBHOOK_URL 環境變數（或為空值）")

    df['global_rank'] = df.reset_index().index + 1

    df_core = df[df['group'] == 'core'].copy()
    df_sub = df[df['group'] == 'sub_thematic'].copy()
    df_macro = df[df['group'] == 'macro_commodity'].copy()

    list_core, rot_core = generate_mobile_list(df_core, prev_history, trade_date)
    list_sub, rot_sub = generate_mobile_list(df_sub, prev_history, trade_date)
    list_macro, rot_macro = generate_mobile_list(df_macro, prev_history, trade_date)

    # 跨象限轉倉快訊最多顯示 12 條（轉強優先），為起漲突破提示預留充裕字元空間
    all_rotations_sorted = sorted(rot_core + rot_sub + rot_macro, key=lambda x: (x[0], x[1]))
    rotation_lines = [item[2] for item in all_rotations_sorted]
    if rotation_lines:
        shown_lines = rotation_lines[:12]
        extra_note = f"\n-# ...另外尚有 {len(rotation_lines) - 12} 檔跨象限變動" if len(rotation_lines) > 12 else ""
        rotation_section = "### 🔄 跨象限轉倉快訊（轉強優先）\n" + "\n".join(shown_lines) + extra_note
    else:
        rotation_section = "### 🔄 跨象限轉倉快訊\n-# 今日暫無板塊突破緩衝區發生轉倉"

    quad_counts = df['quadrant'].value_counts()
    market_breadth_summary = (
        f"### 📈 象限分佈（共 {len(df)} 檔）\n"
        f"🟢 領先 `{quad_counts.get('Leading', 0)}` "
        f"🔵 改善 `{quad_counts.get('Improving', 0)}` "
        f"🟡 轉弱 `{quad_counts.get('Weakening', 0)}` "
        f"🔴 落後 `{quad_counts.get('Lagging', 0)}`"
    )

    spy_benchmark_section = build_spy_benchmark_summary(df, rsp_info, vix_info=None)
    rs_streak_section = build_rs_streak_summary(df)
    vol_sizing_section = build_vol_sizing_summary(df, is_bull_market)
    current_time_iso = datetime.now(timezone.utc).isoformat()

    # 防呆截斷保護，確保單張總覽 Embed 永遠唔會觸及 Discord 4,096 字元上限
    summary_desc = (
        f"{market_regime}\n\n"
        f"{spy_benchmark_section}\n\n"
        f"{market_breadth_summary}\n\n"
        f"{rs_streak_section}\n\n"
        f"{rotation_section}\n\n"
        f"{vol_sizing_section}"
    )
    if len(summary_desc) > 4050:
        summary_desc = summary_desc[:4030] + "\n-# ...(內容過長已截斷)"

    embeds = [
        (
            "Embed 1/4 全市場總覽看板",
            {
                "title": f"🧭 [1/4] 全市場 {len(df)} 檔輪動總結｜大局與配倉",
                "description": summary_desc,
                "color": 5793266
            }
        ),
        (
            "Embed 2/4 核心板塊",
            {
                "title": f"📊 [2/4] 核心板塊與權重行業 ({len(df_core)} 檔)",
                "description": list_core,
                "color": 3066993
            }
        ),
        (
            "Embed 3/4 前沿科技賽道",
            {
                "title": f"🚀 [3/4] 前沿科技與細分賽道 ({len(df_sub)} 檔)",
                "description": list_sub,
                "color": 3447003
            }
        ),
        (
            "Embed 4/4 大宗與宏觀輪動",
            {
                "title": f"🌍 [4/4] 大宗資源與宏觀輪動 ({len(df_macro)} 檔)",
                "description": list_macro,
                "color": 15844367,
                "footer": {
                    "text": f"美股數據日：{trade_date} • EMA標準化RRG (±{RRG_DEADBAND_SIGMA:.2f}σ緩衝) • 🔥連升{RS_STREAK_DAYS}日穿{RS_STREAK_MIN_RATING}"
                },
                "timestamp": current_time_iso
            }
        )
    ]

    for idx, (label, embed) in enumerate(embeds):
        send_single_embed(webhook_url, embed, label)
        if idx < len(embeds) - 1:
            time.sleep(0.8)


# ==========================================
# 5. 主程式入口
# ==========================================
if __name__ == "__main__":
    prev_history = load_previous_history()
    df_result, market_regime, trade_date, is_bull_market, rsp_info, vix_info = calculate_rrg_and_rs(prev_history)
    send_to_discord(df_result, prev_history, market_regime, trade_date, is_bull_market, rsp_info, vix_info)
    save_current_history(df_result, trade_date)
