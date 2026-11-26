"""Event-level recovery measurement and feature engineering."""
import pandas as pd

import numpy as np

from pathlib import Path

from calendar import monthrange

DATE_COL = "OrderDate"

QTY_COL = "Qty"

PROMO_COL_CANDIDATES = ["is_promo_day_v3"]

QTY_PEAK_COL_CANDIDATES = ["Qty_peak_v2", "qty_peak_v2"]

GROUP_COL = "SalePageId"

BASELINE_WINDOW = 14

MAX_K_DAYS = 60

STOP_AT_NEXT_PROMO = True

PROMO_DAYS_CHECK_THRESHOLD = 3

MIN_QTY_PEAK_COUNT_FOR_LONG_PROMO = 3

MIN_PROMO_DAILY_AVG_QTY = 5

DISCOUNT_COLS = [
    "SubtotalPromotionDiscount",
    "SubtotalCouponDiscount",
]

DAY_DISCOUNT_DEPTH_COL = "day_SubtotalPDPrice"

DAY_UNIT_PRICE_COL = "day_UnitPrice"

AVG_DAY_UNIT_PRICE_COL = "avg_day_UnitPrice"

DAY_SALES_AMOUNT_COL = "day_SubtotalSalesAmount"

PROMO_TOTAL_SALES_AMOUNT_COL = "promo_total_sales_amount"

KNOWN_PROMO_FESTIVALS = {
    "10-10": "10/10",
    "11-11": "11/11",
}

PRE_WINDOWS = [7, 14, 28]

PRIOR_PROMO_WINDOWS = [30, 60, 90]

SLOW_RECOVERY_THRESHOLD_DAYS = 7

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def parse_excel_or_text_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = numeric.notna().mean()

    if numeric_ratio > 0.8:
        return pd.to_datetime(
            numeric,
            unit="D",
            origin="1899-12-30",
            errors="coerce"
        )

    return pd.to_datetime(s, errors="coerce")

def resolve_existing_col(df: pd.DataFrame, candidates: list[str], col_description: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(f"Missing {col_description} Candidates: {candidates}")

def read_xlsx_all_sheets(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frames = []

    for sheet_name, temp in sheets.items():
        temp = normalize_columns(temp)

        if temp.empty:
            continue

        temp[GROUP_COL] = str(sheet_name)
        frames.append(temp)

    if not frames:
        raise ValueError("No readable workbook sheets.")

    return pd.concat(frames, ignore_index=True)

def prepare_daily_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    raw = normalize_columns(raw)

    if DATE_COL not in raw.columns:
        raise ValueError(f"Missing date column：{DATE_COL}")

    if QTY_COL not in raw.columns:
        raise ValueError(f"Missing quantity column：{QTY_COL}")

    if GROUP_COL not in raw.columns:
        raise ValueError(f"Missing product column：{GROUP_COL}")

    if DAY_UNIT_PRICE_COL not in raw.columns:
        raise ValueError(f"Missing daily unit-price column：{DAY_UNIT_PRICE_COL}")

    if DAY_SALES_AMOUNT_COL not in raw.columns:
        raise ValueError(f"Missing daily revenue column：{DAY_SALES_AMOUNT_COL}")

    source_promo_col = resolve_existing_col(raw, PROMO_COL_CANDIDATES, "promotion flag")
    source_qty_peak_col = resolve_existing_col(raw, QTY_PEAK_COL_CANDIDATES, "demand peak flag")

    raw[DATE_COL] = parse_excel_or_text_date(raw[DATE_COL])
    raw[QTY_COL] = pd.to_numeric(raw[QTY_COL], errors="coerce").fillna(0)

    raw[DAY_UNIT_PRICE_COL] = pd.to_numeric(raw[DAY_UNIT_PRICE_COL], errors="coerce")
    raw[DAY_SALES_AMOUNT_COL] = pd.to_numeric(raw[DAY_SALES_AMOUNT_COL], errors="coerce").fillna(0)

    raw[source_promo_col] = (
        pd.to_numeric(raw[source_promo_col], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    raw[source_qty_peak_col] = (
        pd.to_numeric(raw[source_qty_peak_col], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    raw["is_promotion"] = raw[source_promo_col]
    raw["Qty_peak_v2"] = raw[source_qty_peak_col]

    for col in DISCOUNT_COLS:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)
        else:
            raw[col] = 0

    if DAY_DISCOUNT_DEPTH_COL in raw.columns:
        raw[DAY_DISCOUNT_DEPTH_COL] = pd.to_numeric(raw[DAY_DISCOUNT_DEPTH_COL], errors="coerce")
    else:
        raw[DAY_DISCOUNT_DEPTH_COL] = (
            -(raw["SubtotalPromotionDiscount"] + raw["SubtotalCouponDiscount"])
            / raw[QTY_COL].replace(0, np.nan)
        )

    raw = raw.dropna(subset=[DATE_COL])

    daily = (
        raw.groupby([GROUP_COL, DATE_COL], as_index=False)
        .agg(
            Qty=(QTY_COL, "sum"),
            is_promotion=("is_promotion", "max"),
            Qty_peak_v2=("Qty_peak_v2", "max"),
            SubtotalPromotionDiscount=("SubtotalPromotionDiscount", "sum"),
            SubtotalCouponDiscount=("SubtotalCouponDiscount", "sum"),
            day_SubtotalPDPrice=(DAY_DISCOUNT_DEPTH_COL, "max"),
            day_UnitPrice=(DAY_UNIT_PRICE_COL, "mean"),
            day_SubtotalSalesAmount=(DAY_SALES_AMOUNT_COL, "sum"),
        )
        .sort_values([GROUP_COL, DATE_COL])
        .reset_index(drop=True)
    )

    return daily, source_promo_col, source_qty_peak_col

def find_clean_baseline(
    g: pd.DataFrame,
    promo_start_pos: int,
    baseline_window: int = BASELINE_WINDOW,
) -> dict:
    before = g.iloc[:promo_start_pos].copy()

    if before.empty:
        return {
            "baseline_qty": np.nan,
            "baseline_start_date": pd.NaT,
            "baseline_end_date": pd.NaT,
            "baseline_method": "no_history",
            "baseline_days": 0,
        }

    for window_end in range(len(before) - 1, baseline_window - 2, -1):
        window_start = window_end - baseline_window + 1
        w = before.iloc[window_start: window_end + 1]

        if len(w) == baseline_window and (w["is_promotion"] == 0).all():
            return {
                "baseline_qty": w["Qty"].mean(),
                "baseline_start_date": w[DATE_COL].min(),
                "baseline_end_date": w[DATE_COL].max(),
                "baseline_method": "clean_consecutive_window",
                "baseline_days": len(w),
            }

    fallback = before[before["is_promotion"] == 0].tail(baseline_window)

    if len(fallback) >= baseline_window:
        return {
            "baseline_qty": fallback["Qty"].mean(),
            "baseline_start_date": fallback[DATE_COL].min(),
            "baseline_end_date": fallback[DATE_COL].max(),
            "baseline_method": "fallback_recent_non_promo_days",
            "baseline_days": len(fallback),
        }

    return {
        "baseline_qty": np.nan,
        "baseline_start_date": pd.NaT,
        "baseline_end_date": pd.NaT,
        "baseline_method": "insufficient_clean_history",
        "baseline_days": len(fallback),
    }

def compute_recent_qty_baseline(g: pd.DataFrame, promo_start_pos: int, window: int) -> float:
    before = g.iloc[:promo_start_pos].copy()

    if len(before) < window:
        return np.nan

    return before.tail(window)["Qty"].mean()

def compute_k_rolling_3day_mean(post_qty: list[float], baseline: float):
    if len(post_qty) < 3:
        return np.nan

    for i in range(len(post_qty)):
        if i >= 2 and np.mean(post_qty[i - 2: i + 1]) >= baseline:
            return i + 1

    return np.nan

def compute_k_consecutive_days(post_qty: list[float], baseline: float, n_days: int):
    if len(post_qty) < n_days:
        return np.nan

    for i in range(len(post_qty)):
        if i >= n_days - 1:
            window = post_qty[i - n_days + 1: i + 1]

            if all(q >= baseline for q in window):
                return i + 1

    return np.nan

def compute_k_cumulative_mean(post_qty: list[float], baseline: float, min_days: int = 3):
    if len(post_qty) < min_days:
        return np.nan

    for i in range(min_days - 1, len(post_qty)):
        cumulative_mean = np.mean(post_qty[: i + 1])

        if cumulative_mean >= baseline:
            return i + 1

    return np.nan

def get_post_promo_window(g: pd.DataFrame, promo_end_pos: int) -> pd.DataFrame:
    post = g.iloc[promo_end_pos + 1: promo_end_pos + 1 + MAX_K_DAYS].copy()

    if STOP_AT_NEXT_PROMO and not post.empty:
        next_promo_positions = np.where(post["is_promotion"].to_numpy() == 1)[0]

        if len(next_promo_positions) > 0:
            post = post.iloc[: next_promo_positions[0]].copy()

    return post

def create_slow_recovery_label(k_value, post_observed_days: int, baseline, threshold_days: int = SLOW_RECOVERY_THRESHOLD_DAYS):
    if pd.isna(baseline) or baseline <= 0:
        return np.nan

    if pd.notna(k_value):
        return int(k_value > threshold_days)

    if post_observed_days >= threshold_days:
        return 1

    return np.nan

def safe_divide(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan

    return numerator / denominator

def compute_trend_slope(values: pd.Series):
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) < 2:
        return np.nan

    x = np.arange(len(values))

    try:
        slope = np.polyfit(x, values.to_numpy(), 1)[0]
    except Exception:
        return np.nan

    return float(slope)

def compute_pre_window_features(g: pd.DataFrame, promo_start_pos: int, windows: list[int] = PRE_WINDOWS) -> dict:
    before = g.iloc[:promo_start_pos].copy()
    features = {}

    for window in windows:
        prefix = f"pre_{window}d_qty"

        if len(before) < window:
            features[f"{prefix}_mean"] = np.nan
            features[f"{prefix}_median"] = np.nan
            features[f"{prefix}_min"] = np.nan
            features[f"{prefix}_max"] = np.nan
            features[f"{prefix}_std"] = np.nan
            features[f"{prefix}_cv"] = np.nan
            features[f"{prefix}_trend_slope"] = np.nan
            features[f"{prefix}_trend_slope_norm"] = np.nan
            continue

        w = before.tail(window)["Qty"].copy()
        mean_value = w.mean()
        std_value = w.std(ddof=0)
        slope = compute_trend_slope(w)

        features[f"{prefix}_mean"] = mean_value
        features[f"{prefix}_median"] = w.median()
        features[f"{prefix}_min"] = w.min()
        features[f"{prefix}_max"] = w.max()
        features[f"{prefix}_std"] = std_value
        features[f"{prefix}_cv"] = safe_divide(std_value, mean_value)
        features[f"{prefix}_trend_slope"] = slope
        features[f"{prefix}_trend_slope_norm"] = safe_divide(slope, mean_value)

    return features

def compute_baseline_change_features(baseline_qty, baseline_7d, baseline_14d) -> dict:
    gap = (
        baseline_7d - baseline_14d
        if pd.notna(baseline_7d) and pd.notna(baseline_14d)
        else np.nan
    )

    return {
        "recent_baseline_ratio_7d_14d": safe_divide(baseline_7d, baseline_14d),
        "baseline_gap_7d_minus_14d": gap,
        "baseline_growth_rate_7d_vs_14d": safe_divide(gap, baseline_14d),
        "baseline_vs_clean_baseline_ratio": safe_divide(baseline_14d, baseline_qty),
    }

def count_promo_events_in_window(window_df: pd.DataFrame) -> int:
    if window_df.empty:
        return 0

    promo = window_df["is_promotion"].fillna(0).astype(int)
    starts = (promo == 1) & (promo.shift(fill_value=0) == 0)

    return int(starts.sum())

def compute_prior_promo_features(
    g: pd.DataFrame,
    promo_start_pos: int,
    promo_start_date,
    windows: list[int] = PRIOR_PROMO_WINDOWS,
) -> dict:
    before = g.iloc[:promo_start_pos].copy()
    features = {}

    if before.empty:
        features["days_since_last_promo_day"] = np.nan
    else:
        previous_promo_days = before[before["is_promotion"] == 1]

        if previous_promo_days.empty:
            features["days_since_last_promo_day"] = np.nan
        else:
            last_promo_date = previous_promo_days[DATE_COL].max()
            features["days_since_last_promo_day"] = (
                pd.Timestamp(promo_start_date) - pd.Timestamp(last_promo_date)
            ).days

    for window in windows:
        start_date = pd.Timestamp(promo_start_date) - pd.Timedelta(days=window)

        w = before[
            (before[DATE_COL] >= start_date)
            & (before[DATE_COL] < pd.Timestamp(promo_start_date))
        ].copy()

        features[f"prior_promo_day_count_{window}d"] = int(w["is_promotion"].sum()) if not w.empty else 0
        features[f"prior_promo_event_count_{window}d"] = count_promo_events_in_window(w)

    return features

def is_near_payday(day: int) -> int:
    paydays = [5, 10, 25]
    return int(any(abs(day - payday) <= 2 for payday in paydays))

def compute_time_features(promo_start_date, promo_end_date) -> dict:
    start = pd.Timestamp(promo_start_date)
    end = pd.Timestamp(promo_end_date)

    date_range = pd.date_range(start=start, end=end, freq="D")
    weekend_days = int(sum(d.weekday() >= 5 for d in date_range))
    promo_days = len(date_range)
    month_end_day = monthrange(start.year, start.month)[1]

    return {
        "promo_start_month": start.month,
        "promo_start_day": start.day,
        "promo_start_weekday": start.weekday(),
        "promo_start_is_weekend": int(start.weekday() >= 5),
        "promo_end_weekday": end.weekday(),
        "promo_end_is_weekend": int(end.weekday() >= 5),
        "contains_weekend": int(weekend_days > 0),
        "weekend_days": weekend_days,
        "weekend_ratio": safe_divide(weekend_days, promo_days),
        "promo_start_is_month_start": int(start.day <= 5),
        "promo_start_is_month_end": int(start.day >= month_end_day - 4),
        "promo_start_days_to_month_end": month_end_day - start.day,
        "promo_start_is_near_payday": is_near_payday(start.day),
    }

def check_known_promo_festival(promo_start_date, promo_end_date) -> tuple[int, str]:
    if pd.isna(promo_start_date) or pd.isna(promo_end_date):
        return 0, ""

    date_range = pd.date_range(
        start=pd.Timestamp(promo_start_date),
        end=pd.Timestamp(promo_end_date),
        freq="D",
    )

    matched_festivals = []

    for d in date_range:
        mmdd = d.strftime("%m-%d")

        if mmdd in KNOWN_PROMO_FESTIVALS:
            matched_festivals.append(KNOWN_PROMO_FESTIVALS[mmdd])

    matched_festivals = sorted(set(matched_festivals))

    return int(len(matched_festivals) > 0), ",".join(matched_festivals)

def apply_promo_filter_rules(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    g["is_promotion_original"] = g["is_promotion"].copy()
    g["is_removed_by_peak_rule"] = 0
    g["is_removed_by_avg_qty_rule"] = 0
    g["is_removed_by_any_rule"] = 0
    g["removed_promo_episode_id"] = np.nan
    g["removed_reason"] = ""

    g["original_promo_event_count_for_group"] = 0
    g["removed_promo_event_count_for_group"] = 0
    g["removed_by_peak_rule_event_count_for_group"] = 0
    g["removed_by_avg_qty_rule_event_count_for_group"] = 0
    g["removed_by_both_rules_event_count_for_group"] = 0
    g["removed_promo_day_count_for_group"] = 0
    g["retained_promo_event_count_for_group"] = 0

    g["promo_episode_tmp"] = (
        g["is_promotion_original"] != g["is_promotion_original"].shift()
    ).cumsum()

    original_blocks = (
        g[g["is_promotion_original"] == 1]
        .groupby("promo_episode_tmp")
        .agg(
            promo_start_pos=(DATE_COL, lambda x: x.index.min()),
            promo_end_pos=(DATE_COL, lambda x: x.index.max()),
            promo_days=(DATE_COL, "count"),
            qty_peak_v2_count=("Qty_peak_v2", "sum"),
            promo_qty_total=("Qty", "sum"),
        )
        .reset_index()
    )

    original_promo_event_count = len(original_blocks)
    removed_promo_event_count = 0
    removed_by_peak_rule_event_count = 0
    removed_by_avg_qty_rule_event_count = 0
    removed_by_both_rules_event_count = 0
    removed_promo_day_count = 0

    for _, block in original_blocks.iterrows():
        promo_days = int(block["promo_days"])
        qty_peak_v2_count = int(block["qty_peak_v2_count"])
        promo_qty_total = float(block["promo_qty_total"])
        promo_start_pos = int(block["promo_start_pos"])
        promo_end_pos = int(block["promo_end_pos"])
        original_episode_id = block["promo_episode_tmp"]

        promo_daily_avg_qty = (
            promo_qty_total / promo_days
            if promo_days > 0
            else np.nan
        )

        fail_peak_rule = (
            promo_days > PROMO_DAYS_CHECK_THRESHOLD
            and qty_peak_v2_count < MIN_QTY_PEAK_COUNT_FOR_LONG_PROMO
        )

        fail_avg_qty_rule = (
            pd.notna(promo_daily_avg_qty)
            and promo_daily_avg_qty < MIN_PROMO_DAILY_AVG_QTY
        )

        should_remove = fail_peak_rule or fail_avg_qty_rule

        if should_remove:
            reasons = []

            if fail_peak_rule:
                reasons.append("peak_rule")
                removed_by_peak_rule_event_count += 1

            if fail_avg_qty_rule:
                reasons.append("avg_qty_rule")
                removed_by_avg_qty_rule_event_count += 1

            if fail_peak_rule and fail_avg_qty_rule:
                removed_by_both_rules_event_count += 1

            removed_reason = "+".join(reasons)

            g.loc[promo_start_pos:promo_end_pos, "is_promotion"] = 0
            g.loc[promo_start_pos:promo_end_pos, "is_removed_by_any_rule"] = 1
            g.loc[promo_start_pos:promo_end_pos, "is_removed_by_peak_rule"] = int(fail_peak_rule)
            g.loc[promo_start_pos:promo_end_pos, "is_removed_by_avg_qty_rule"] = int(fail_avg_qty_rule)
            g.loc[promo_start_pos:promo_end_pos, "removed_promo_episode_id"] = original_episode_id
            g.loc[promo_start_pos:promo_end_pos, "removed_reason"] = removed_reason

            removed_promo_event_count += 1
            removed_promo_day_count += promo_days

    retained_promo_event_count = original_promo_event_count - removed_promo_event_count

    g["original_promo_event_count_for_group"] = original_promo_event_count
    g["removed_promo_event_count_for_group"] = removed_promo_event_count
    g["removed_by_peak_rule_event_count_for_group"] = removed_by_peak_rule_event_count
    g["removed_by_avg_qty_rule_event_count_for_group"] = removed_by_avg_qty_rule_event_count
    g["removed_by_both_rules_event_count_for_group"] = removed_by_both_rules_event_count
    g["removed_promo_day_count_for_group"] = removed_promo_day_count
    g["retained_promo_event_count_for_group"] = retained_promo_event_count

    g = g.drop(columns=["promo_episode_tmp"])

    return g

def calculate_events_for_group(g: pd.DataFrame, group_value) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = g.sort_values(DATE_COL).reset_index(drop=True).copy()

    g = apply_promo_filter_rules(g)

    g["promo_episode_tmp"] = (
        g["is_promotion"] != g["is_promotion"].shift()
    ).cumsum()

    promo_blocks = (
        g[g["is_promotion"] == 1]
        .groupby("promo_episode_tmp")
        .agg(
            promo_start_date=(DATE_COL, "min"),
            promo_end_date=(DATE_COL, "max"),
            promo_start_pos=(DATE_COL, lambda x: x.index.min()),
            promo_end_pos=(DATE_COL, lambda x: x.index.max()),
            promo_days=(DATE_COL, "count"),
            qty_peak_v2_count=("Qty_peak_v2", "sum"),
            promo_qty_total=("Qty", "sum"),
            promo_period_max_demand=("Qty", "max"),
            promo_total_sales_amount=("day_SubtotalSalesAmount", "sum"),
            subtotal_promotion_discount=("SubtotalPromotionDiscount", "sum"),
            subtotal_coupon_discount=("SubtotalCouponDiscount", "sum"),
            max_day_SubtotalPDPrice=("day_SubtotalPDPrice", "max"),
            avg_day_UnitPrice=("day_UnitPrice", "mean"),
        )
        .reset_index(drop=True)
    )

    event_rows = []

    g["promo_event_id"] = pd.Series([None] * len(g), dtype=object, index=g.index)
    g["K_rolling_3day_mean"] = np.nan
    g["K_consecutive_2days"] = np.nan
    g["K_consecutive_3days"] = np.nan
    g["K_cumulative_mean"] = np.nan
    g["baseline_qty_for_event"] = np.nan

    for local_event_no, ep in promo_blocks.iterrows():
        promo_start_pos = int(ep["promo_start_pos"])
        promo_end_pos = int(ep["promo_end_pos"])

        event_id = f"{group_value}_{local_event_no + 1:03d}"

        baseline_info = find_clean_baseline(g, promo_start_pos)
        baseline = baseline_info["baseline_qty"]

        baseline_14d_qty = compute_recent_qty_baseline(g, promo_start_pos, window=14)
        baseline_7d_qty = compute_recent_qty_baseline(g, promo_start_pos, window=7)

        post = get_post_promo_window(g, promo_end_pos)
        post_qty = post["Qty"].tolist()

        if pd.notna(baseline) and baseline > 0 and len(post_qty) > 0:
            k_rolling_3day_mean = compute_k_rolling_3day_mean(post_qty, baseline)
            k_consecutive_2days = compute_k_consecutive_days(post_qty, baseline, n_days=2)
            k_consecutive_3days = compute_k_consecutive_days(post_qty, baseline, n_days=3)
            k_cumulative_mean = compute_k_cumulative_mean(post_qty, baseline, min_days=3)
        else:
            k_rolling_3day_mean = np.nan
            k_consecutive_2days = np.nan
            k_consecutive_3days = np.nan
            k_cumulative_mean = np.nan

        is_known_promo_festival, known_promo_festival_name = check_known_promo_festival(
            ep["promo_start_date"],
            ep["promo_end_date"],
        )

        promo_discount_abs = abs(ep["subtotal_promotion_discount"])
        coupon_discount_abs = abs(ep["subtotal_coupon_discount"])
        total_discount_abs = promo_discount_abs + coupon_discount_abs

        discount_mix_promotion = (
            promo_discount_abs / total_discount_abs
            if total_discount_abs > 0
            else np.nan
        )

        avg_promotion_discount_per_qty = (
            promo_discount_abs / ep["promo_qty_total"]
            if ep["promo_qty_total"] > 0
            else np.nan
        )

        avg_coupon_discount_per_qty = (
            coupon_discount_abs / ep["promo_qty_total"]
            if ep["promo_qty_total"] > 0
            else np.nan
        )

        avg_day_SubtotalPDPrice = (
            total_discount_abs / ep["promo_qty_total"]
            if ep["promo_qty_total"] > 0
            else np.nan
        )

        pre_window_features = compute_pre_window_features(g, promo_start_pos)
        baseline_change_features = compute_baseline_change_features(
            baseline,
            baseline_7d_qty,
            baseline_14d_qty,
        )
        prior_promo_features = compute_prior_promo_features(
            g,
            promo_start_pos,
            ep["promo_start_date"],
        )
        time_features = compute_time_features(
            ep["promo_start_date"],
            ep["promo_end_date"],
        )

        post_observed_days = len(post)

        row = {
            GROUP_COL: group_value,
            "promo_event_id": event_id,

            "promo_start_date": ep["promo_start_date"],
            "promo_end_date": ep["promo_end_date"],
            "promo_days": ep["promo_days"],
            "qty_peak_v2_count": ep["qty_peak_v2_count"],

            "is_known_promo_festival": is_known_promo_festival,
            "known_promo_festival_name": known_promo_festival_name,

            "baseline_qty": baseline,
            "Baseline_14d_qty": baseline_14d_qty,
            "Baseline_7d_qty": baseline_7d_qty,
            "baseline_start_date": baseline_info["baseline_start_date"],
            "baseline_end_date": baseline_info["baseline_end_date"],
            "baseline_method": baseline_info["baseline_method"],
            "baseline_days": baseline_info["baseline_days"],

            "post_observed_days": post_observed_days,

            "K_rolling_3day_mean": k_rolling_3day_mean,
            "K_consecutive_2days": k_consecutive_2days,
            "K_consecutive_3days": k_consecutive_3days,
            "K_cumulative_mean": k_cumulative_mean,

            "slow_recovery_7d_rolling_3day_mean": create_slow_recovery_label(
                k_rolling_3day_mean,
                post_observed_days,
                baseline,
            ),
            "slow_recovery_7d_consecutive_2days": create_slow_recovery_label(
                k_consecutive_2days,
                post_observed_days,
                baseline,
            ),
            "slow_recovery_7d_consecutive_3days": create_slow_recovery_label(
                k_consecutive_3days,
                post_observed_days,
                baseline,
            ),
            "slow_recovery_7d_cumulative_mean": create_slow_recovery_label(
                k_cumulative_mean,
                post_observed_days,
                baseline,
            ),

            "promo_qty_total": ep["promo_qty_total"],
            "promo_period_max_demand": ep["promo_period_max_demand"],
            "promo_total_sales_amount": ep["promo_total_sales_amount"],

            "subtotal_promotion_discount_abs": promo_discount_abs,
            "subtotal_coupon_discount_abs": coupon_discount_abs,
            "total_discount_abs": total_discount_abs,

            "discount_mix_promotion": discount_mix_promotion,

            "avg_promotion_discount_per_qty": avg_promotion_discount_per_qty,
            "avg_coupon_discount_per_qty": avg_coupon_discount_per_qty,
            "avg_day_SubtotalPDPrice": avg_day_SubtotalPDPrice,
            "max_day_SubtotalPDPrice": ep["max_day_SubtotalPDPrice"],

            "avg_day_UnitPrice": ep["avg_day_UnitPrice"],
        }

        row.update(pre_window_features)
        row.update(baseline_change_features)
        row.update(prior_promo_features)
        row.update(time_features)

        event_rows.append(row)

        g.loc[promo_start_pos:promo_end_pos, "promo_event_id"] = event_id
        g.loc[promo_end_pos, "baseline_qty_for_event"] = baseline
        g.loc[promo_end_pos, "K_rolling_3day_mean"] = k_rolling_3day_mean
        g.loc[promo_end_pos, "K_consecutive_2days"] = k_consecutive_2days
        g.loc[promo_end_pos, "K_consecutive_3days"] = k_consecutive_3days
        g.loc[promo_end_pos, "K_cumulative_mean"] = k_cumulative_mean

    events = pd.DataFrame(event_rows)
    daily_marked = g.drop(columns=["promo_episode_tmp"])

    return events, daily_marked

def calculate_all_k(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_events = []
    all_daily = []

    for group_value, g in daily.groupby(GROUP_COL, sort=False):
        events, daily_marked = calculate_events_for_group(g, group_value)

        if not events.empty:
            all_events.append(events)

        all_daily.append(daily_marked)

    event_result = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    daily_result = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()

    if not daily_result.empty:
        original_event_count_before_rule = (
            daily_result.groupby(GROUP_COL)["original_promo_event_count_for_group"].max().sum()
        )
        removed_event_count_by_rule = (
            daily_result.groupby(GROUP_COL)["removed_promo_event_count_for_group"].max().sum()
        )
        removed_by_peak_rule_event_count = (
            daily_result.groupby(GROUP_COL)["removed_by_peak_rule_event_count_for_group"].max().sum()
        )
        removed_by_avg_qty_rule_event_count = (
            daily_result.groupby(GROUP_COL)["removed_by_avg_qty_rule_event_count_for_group"].max().sum()
        )
        removed_by_both_rules_event_count = (
            daily_result.groupby(GROUP_COL)["removed_by_both_rules_event_count_for_group"].max().sum()
        )
        retained_event_count_after_rule = (
            daily_result.groupby(GROUP_COL)["retained_promo_event_count_for_group"].max().sum()
        )

        removed_day_count_by_rule = int(daily_result["is_removed_by_any_rule"].sum())
        removed_day_count_by_peak_rule = int(daily_result["is_removed_by_peak_rule"].sum())
        removed_day_count_by_avg_qty_rule = int(daily_result["is_removed_by_avg_qty_rule"].sum())

        print(f"Original event count: {int(original_event_count_before_rule)}")
        print(f"Excluded events: {int(removed_event_count_by_rule)}")
        print(f"Events excluded by peak rule: {int(removed_by_peak_rule_event_count)}")
        print(f"Events excluded by average-quantity rule (< {MIN_PROMO_DAILY_AVG_QTY}): {int(removed_by_avg_qty_rule_event_count)}")
        print(f"Events excluded by both rules: {int(removed_by_both_rules_event_count)}")
        print(f"Retained events: {int(retained_event_count_after_rule)}")
        print(f"Excluded promotion days: {removed_day_count_by_rule}")
        print(f"Days excluded by peak rule: {removed_day_count_by_peak_rule}")
        print(f"Days excluded by average-quantity rule (< {MIN_PROMO_DAILY_AVG_QTY}): {removed_day_count_by_avg_qty_rule}")

    if not event_result.empty:
        total_event_count = len(event_result)
        k_cumulative_missing_count = event_result["K_cumulative_mean"].isna().sum()
        k_cumulative_available_count = event_result["K_cumulative_mean"].notna().sum()

        print(f"Output events: {total_event_count}")
        print(f"Missing cumulative K: {k_cumulative_missing_count}")
        print(f"Observed cumulative K: {k_cumulative_available_count}")

    if event_result.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            event_result.groupby(GROUP_COL, as_index=False)
            .agg(
                promo_event_count=("promo_event_id", "count"),

                avg_K_rolling_3day_mean=("K_rolling_3day_mean", "mean"),
                avg_K_consecutive_2days=("K_consecutive_2days", "mean"),
                avg_K_consecutive_3days=("K_consecutive_3days", "mean"),
                avg_K_cumulative_mean=("K_cumulative_mean", "mean"),

                slow_recovery_7d_consecutive_2days_count=("slow_recovery_7d_consecutive_2days", "sum"),

                avg_baseline_qty=("baseline_qty", "mean"),
                avg_Baseline_14d_qty=("Baseline_14d_qty", "mean"),
                avg_Baseline_7d_qty=("Baseline_7d_qty", "mean"),
                avg_promo_days=("promo_days", "mean"),
                avg_qty_peak_v2_count=("qty_peak_v2_count", "mean"),

                avg_day_UnitPrice=("avg_day_UnitPrice", "mean"),
                avg_promo_total_sales_amount=("promo_total_sales_amount", "mean"),

                known_promo_festival_event_count=("is_known_promo_festival", "sum"),

                avg_pre_7d_qty_mean=("pre_7d_qty_mean", "mean"),
                avg_pre_14d_qty_mean=("pre_14d_qty_mean", "mean"),
                avg_pre_28d_qty_mean=("pre_28d_qty_mean", "mean"),

                avg_prior_promo_day_count_30d=("prior_promo_day_count_30d", "mean"),
                avg_days_since_last_promo_day=("days_since_last_promo_day", "mean"),
            )
            .sort_values("promo_event_count", ascending=False)
        )

    return event_result, daily_result, summary
