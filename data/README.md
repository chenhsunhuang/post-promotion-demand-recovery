# Data Requirements

The enterprise dataset and processed event workbooks are not distributed. Use only authorized local data. Place input tables in `data/private/`, which is excluded from version control.

## Product-day input

Expected columns: `SalePageId`, `OrderDate`, `Qty`, `SubtotalPromotionDiscount`, `SubtotalCouponDiscount`, `day_UnitPrice`, and `day_SubtotalSalesAmount`. The last two fields are required for event feature and revenue calculations. The input must contain at most one row per product and date. Notebook 01 demonstrates detection with fictional data; its output can be passed to the recovery pipeline in Notebook 02.

## Category event inputs

| Local file | Sheet |
|---|---|
| `feminine_hygiene_events.xlsx` | `promotion_events_K` |
| `sunscreen_events.xlsx` | `promotion_events_K` |
| `facial_masks_events.xlsx` | `promotion_events_K` |

Each table requires `K_consecutive_2days` and the category-specific feature columns specified in [`src/model_config.py`](../src/model_config.py). For the exploratory revenue model, the original target calculation uses `promo_total_sales_amount` when present, `avg_day_baseline_sales_amount` when available, and otherwise historical proxy fields. No event-level predictions or data should be committed.
