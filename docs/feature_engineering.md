# Feature Engineering

The team aggregated historical transaction records by product and date after joining product-category information. Category labels were assigned using product-name keywords. The event-level pipeline derives the following features:

| Feature group | Examples |
|---|---|
| Historical demand | 7-, 14-, 28-day mean, median, standard deviation, CV and trend slope |
| Baseline behavior | Clean 14-day non-promotion baseline, recent demand ratios and growth |
| Promotion history | Prior event and promotion-day counts over 30, 60 and 90 days |
| Calendar | Month, weekday, weekends, month boundaries, payday proximity and known shopping festivals |
| Event characteristics | Duration, average and maximum discount depth, first-day demand and average unit price |

The historical features are defined from dates before the event. Realized event characteristics are used for scenario analysis or updated after the relevant information becomes available. Exact feature lists and category thresholds are in [`src/model_config.py`](../src/model_config.py).
