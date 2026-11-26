"""Multi-stage rolling promotion detection."""
import numpy as np
import pandas as pd

COL_ID   = 'SalePageId'

COL_DATE = 'OrderDate'

COL_PD   = 'SubtotalPromotionDiscount'

COL_CD   = 'SubtotalCouponDiscount'

COL_QTY  = 'Qty'

MAD_SCALE     = 1.4826

WINDOW        = 14

MULTIPLIER_P0 = 1.5

MULTIPLIER_P1 = 1.0

MULTIPLIER_P2 = 1.5

MULTIPLIER_P3 = 1.5

def phase0_flags(series: np.ndarray, window: int, multiplier: float) -> np.ndarray:
    n     = len(series)
    flags = np.zeros(n, dtype=int)
    init  = series[:window]
    valid = init[~np.isnan(init)]
    if len(valid) >= 2:
        med = np.median(valid)
        mad = np.median(np.abs(valid - med))
        threshold = med + multiplier * MAD_SCALE * mad
        for k in range(min(window, n)):
            if not np.isnan(series[k]) and series[k] > threshold:
                flags[k] = 1
    return flags

def twophase_with_p0(prod_df: pd.DataFrame) -> pd.DataFrame:
    prod_df = prod_df.sort_values(COL_DATE).reset_index(drop=True)
    n    = len(prod_df)
    disc = prod_df['day_SubtotalPDPrice'].to_numpy(dtype=float)
    qty  = prod_df[COL_QTY].to_numpy(dtype=float)

    is_v0    = phase0_flags(disc, WINDOW, MULTIPLIER_P0)
    is_pk_v0 = phase0_flags(qty,  WINDOW, MULTIPLIER_P0)


    avg_p1 = np.full(n, np.nan); std_p1 = np.full(n, np.nan); is_v1 = np.zeros(n, dtype=int)
    for i in range(n):
        if i < WINDOW: continue
        w_disc = disc[i - WINDOW:i]; w_v0 = is_v0[i - WINDOW:i]
        vals = w_disc[(w_v0 == 0) & ~np.isnan(w_disc)]
        if len(vals) < 2: vals = w_disc[~np.isnan(w_disc)]
        if len(vals) < 2: continue
        avg_p1[i] = vals.mean(); std_p1[i] = vals.std(ddof=1)
        if not np.isnan(disc[i]) and disc[i] > avg_p1[i] + MULTIPLIER_P1 * std_p1[i]:
            is_v1[i] = 1


    avg_p2 = np.full(n, np.nan); std_p2 = np.full(n, np.nan); is_v2 = np.zeros(n, dtype=int)
    for i in range(n):
        wd, j = [], i - 1
        while len(wd) < WINDOW and j >= 0:
            if is_v0[j] == 0 and is_v1[j] == 0: wd.append(disc[j])
            j -= 1
        if len(wd) < WINDOW: continue
        arr = np.array(wd); avg_p2[i] = arr.mean(); std_p2[i] = arr.std(ddof=1)
        if not np.isnan(disc[i]) and disc[i] > avg_p2[i] + MULTIPLIER_P2 * std_p2[i]:
            is_v2[i] = 1


    avg_qty_p1 = np.full(n, np.nan); std_qty_p1 = np.full(n, np.nan); is_pk_v1 = np.zeros(n, dtype=int)
    for i in range(n):
        if i < WINDOW: continue
        w_qty = qty[i - WINDOW:i]; w_v0 = is_pk_v0[i - WINDOW:i]
        vals = w_qty[(w_v0 == 0) & ~np.isnan(w_qty)]
        if len(vals) < 2: vals = w_qty[~np.isnan(w_qty)]
        if len(vals) < 2: continue
        avg_qty_p1[i] = vals.mean(); std_qty_p1[i] = vals.std(ddof=1)
        if not np.isnan(qty[i]) and qty[i] > avg_qty_p1[i] + MULTIPLIER_P1 * std_qty_p1[i]:
            is_pk_v1[i] = 1


    avg_qty_p2 = np.full(n, np.nan); std_qty_p2 = np.full(n, np.nan); is_pk_v2 = np.zeros(n, dtype=int)
    for i in range(n):
        wq, j = [], i - 1
        while len(wq) < WINDOW and j >= 0:
            if is_pk_v0[j] == 0 and is_pk_v1[j] == 0: wq.append(qty[j])
            j -= 1
        if len(wq) < WINDOW: continue
        arr = np.array(wq); avg_qty_p2[i] = arr.mean(); std_qty_p2[i] = arr.std(ddof=1)
        if not np.isnan(qty[i]) and qty[i] > avg_qty_p2[i] + MULTIPLIER_P2 * std_qty_p2[i]:
            is_pk_v2[i] = 1


    avg_p3 = np.full(n, np.nan); std_p3 = np.full(n, np.nan); is_v3 = np.zeros(n, dtype=int)
    for i in range(n):
        wd, j = [], i - 1
        while len(wd) < WINDOW and j >= 0:
            if is_v0[j] == 0 and is_v1[j] == 0 and is_v2[j] == 0: wd.append(disc[j])
            j -= 1
        if len(wd) < WINDOW: continue
        arr = np.array(wd); avg_p3[i] = arr.mean(); std_p3[i] = arr.std(ddof=1)
        if not np.isnan(disc[i]) and disc[i] > avg_p3[i] + MULTIPLIER_P3 * std_p3[i]:
            is_v3[i] = 1

    prod_df['is_promo_v0']     = is_v0
    prod_df['avg_p1']          = avg_p1;  prod_df['std_p1']     = std_p1;  prod_df['is_promo_day_v1'] = is_v1
    prod_df['avg_p2']          = avg_p2;  prod_df['std_p2']     = std_p2;  prod_df['is_promo_day_v2'] = is_v2
    prod_df['avg_p3']          = avg_p3;  prod_df['std_p3']     = std_p3;  prod_df['is_promo_day_v3'] = is_v3
    prod_df['is_peak_v0']      = is_pk_v0
    prod_df['avg_qty_p1']      = avg_qty_p1; prod_df['std_qty_p1'] = std_qty_p1; prod_df['Qty_peak_v1'] = is_pk_v1
    prod_df['avg_qty_p2']      = avg_qty_p2; prod_df['std_qty_p2'] = std_qty_p2; prod_df['Qty_peak_v2'] = is_pk_v2
    return prod_df

def _find_blocks(arr):
    blocks, i, n = [], 0, len(arr)
    while i < n:
        if arr[i] == 1:
            j = i + 1
            while j < n and arr[j] == 1: j += 1
            blocks.append((i, j - 1))
            i = j
        else:
            i += 1
    return blocks

def _remove_zero_peak_blocks(arr, pk2):
    i, n = 0, len(arr)
    while i < n:
        if arr[i] == 1:
            j = i + 1
            while j < n and arr[j] == 1: j += 1
            if pk2[i:j].sum() == 0: arr[i:j] = 0
            i = j
        else:
            i += 1

def _filter_blocks_by_peak(arr, pk2):
    """Filter promotion blocks using the minimum number of demand peaks."""
    i, n = 0, len(arr)
    while i < n:
        if arr[i] == 1:
            j = i + 1
            while j < n and arr[j] == 1: j += 1
            block_len = j - i
            pk_sum    = pk2[i:j].sum()
            if block_len > 3:
                if pk_sum < 3: arr[i:j] = 0
            else:
                if pk_sum < 1: arr[i:j] = 0
            i = j
        else:
            i += 1

def _fill_gaps(arr, max_gap=2):
    blocks = _find_blocks(arr)
    for k in range(len(blocks) - 1):
        gs = blocks[k][1] + 1
        ge = blocks[k + 1][0] - 1
        if ge - gs + 1 <= max_gap:
            arr[gs:ge + 1] = 1

def _trim_zero_pd_boundaries(arr, pd_arr):
    i, n = 0, len(arr)
    while i < n:
        if arr[i] == 1:
            j = i + 1
            while j < n and arr[j] == 1: j += 1
            while i < j and pd_arr[i] == 0: arr[i] = 0; i += 1
            end = j - 1
            while end >= i and pd_arr[end] == 0: arr[end] = 0; end -= 1
            i = j
        else:
            i += 1

def postprocess_promo(df: pd.DataFrame) -> pd.DataFrame:
    v2  = df['is_promo_day_v2'].values.copy()
    v3  = df['is_promo_day_v3'].values.copy()
    pk2 = df['Qty_peak_v2'].values
    pd_arr = (df[COL_PD] + df[COL_CD]).abs().values


    _remove_zero_peak_blocks(v2, pk2)
    _fill_gaps(v2)


    v3[v2 == 1] = 1
    v3_src = v3.copy()
    for i in range(len(v3)):
        if v3_src[i] == 1:
            prev_ok = (i == 0) or (v3_src[i - 1] == 0)
            next_ok = (i == len(v3) - 1) or (v3_src[i + 1] == 0)
            if prev_ok and next_ok:
                v3[i] = 0
    _fill_gaps(v3)
    _trim_zero_pd_boundaries(v3, pd_arr)
    _filter_blocks_by_peak(v3, pk2)


    v3_src2 = v3.copy()
    for i in range(len(v3)):
        if v3_src2[i] == 1:
            prev_ok = (i == 0) or (v3_src2[i - 1] == 0)
            next_ok = (i == len(v3) - 1) or (v3_src2[i + 1] == 0)
            if prev_ok and next_ok:
                v3[i] = 0

    result = df.copy()
    result['is_promo_day_v2'] = v2
    result['is_promo_day_v3'] = v3
    return result

def detect_promotions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the product-level detection pipeline to one category's daily records.

    Missing-date quantity/discount values are interpolated for event detection;
    imputed quantities are restored to zero afterward. Other input columns are retained.
    """
    df = df.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    if df.empty:
        raise ValueError("Input daily data are empty.")
    if df.duplicated([COL_ID, COL_DATE]).any():
        raise ValueError("Expected one row per product and date.")

    df['day_SubtotalPDPrice'] = (
        (df[COL_PD] + df[COL_CD]).abs() / df[COL_QTY].replace(0, np.nan)
    )


    product_ids     = sorted(df[COL_ID].unique())
    global_min_date = df[COL_DATE].min()
    global_max_date = df[COL_DATE].max()
    full_date_range = pd.date_range(global_min_date, global_max_date, freq='D')

    filled_parts = []
    for pid in product_ids:
        prod = df[df[COL_ID] == pid].set_index(COL_DATE)
        prod = prod.reindex(full_date_range)
        prod.index.name = COL_DATE
        prod = prod.reset_index()
        prod[COL_ID] = pid

        prod['_is_filled'] = prod[COL_QTY].isna()
        fill_mask = prod['_is_filled']


        orig_qty = prod[COL_QTY].where(~fill_mask)
        pq = orig_qty.ffill()[fill_mask]; nq = orig_qty.bfill()[fill_mask]
        both_q = pq.notna() & nq.notna()
        interp_qty = pd.Series(np.nan, index=pq.index)
        interp_qty[both_q]              = (pq[both_q] + nq[both_q]) / 2
        interp_qty[pq.notna() & nq.isna()] = pq[pq.notna() & nq.isna()]
        interp_qty[pq.isna() & nq.notna()] = nq[pq.isna() & nq.notna()]
        prod.loc[fill_mask, COL_QTY] = interp_qty


        orig_pd = prod['day_SubtotalPDPrice'].where(~fill_mask)
        p = orig_pd.ffill()[fill_mask]; n = orig_pd.bfill()[fill_mask]
        both = p.notna() & n.notna()
        interp = pd.Series(np.nan, index=p.index)
        interp[both]               = (p[both] + n[both]) / 2
        interp[p.notna() & n.isna()] = p[p.notna() & n.isna()]
        interp[p.isna() & n.notna()] = n[p.isna() & n.notna()]
        prod.loc[fill_mask, 'day_SubtotalPDPrice'] = interp

        for c in [COL_PD, COL_CD]:
            prod[c] = prod[c].fillna(0)

        filled_parts.append(prod)

    daily = pd.concat(filled_parts, ignore_index=True)
    daily = daily.sort_values([COL_ID, COL_DATE]).reset_index(drop=True)


    results = []
    for pid in product_ids:
        result = postprocess_promo(twophase_with_p0(daily[daily[COL_ID] == pid].copy()))
        result['Qty_Impute'] = np.where(result['_is_filled'], result[COL_QTY], np.nan)
        result.loc[result['_is_filled'], COL_QTY] = 0
        results.append(result)
    return pd.concat(results, ignore_index=True)
