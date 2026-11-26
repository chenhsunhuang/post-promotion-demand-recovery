# Methodology

## Promotion Event Detection

Daily discount depth is the absolute sum of promotional and coupon discounts divided by quantity sold. The original algorithm fills missing dates using the average of adjacent observed values (or the nearest available value) for detection, then restores imputed quantities to zero.

Initialization flags unusual observations in the first 14 days using the median plus 1.5 scaled MADs (scale factor 1.4826). Phase 1 applies a rolling mean plus one standard deviation; phases 2 and 3 use the latest 14 previously unflagged observations with a 1.5-standard-deviation threshold. Demand peaks are detected with a corresponding two-stage rolling process. Post-processing merges gaps of at most two days, removes isolated days, trims zero-discount boundaries and filters blocks by demand-peak support.

## Recovery Time

Contiguous flagged days define events. Events longer than three days require at least three demand-peak days, and events averaging fewer than five units per day are excluded. The baseline is the latest clean consecutive 14-day window before a promotion, or the latest 14 non-promotion days when a consecutive window is unavailable. Events with no valid baseline or no observed recovery have missing K.

The alternatives use cumulative mean (minimum three days), three-day moving mean, two consecutive days, or three consecutive days at or above baseline. K records the day on which the criterion is confirmed. Observation ends after 60 days or at the next promotion. The two-consecutive-day definition was selected to balance stability and sample retention.

## Category-Specific Modeling

The original experiments compared Logistic Regression, RBF SVM, KNN, Random Forest, and XGBoost on an 80/20 stratified split. Class weights are used for Logistic Regression, SVM and Random Forest; XGBoost uses balanced sample weights. KNN uses five neighbors and distance-weighted voting. Selected-model specifications use median imputation, log transformations of baseline and first-day demand, and standardization. The forward-selection procedure in the available feminine hygiene and facial mask notebooks visits candidate features sequentially and retains features improving macro F1 on the evaluation split.

| Category | Low K | Medium K | High K |
|---|---|---|---|
| Feminine hygiene | 1–4 | 5–8 | >8 |
| Sunscreen | 1–2 | 3–5 | >5 |
| Facial masks | 1–3 | 4–6 | >6 |

Permutation importance and Spearman correlations provide descriptive feature diagnostics. The original imputation precedes the split, and feature selection uses the evaluation split; the reported scores describe the historical experiment, not an independent prospective test. Missing K values are excluded, so performance pertains to events with observed recovery. Promotion flags and some features use contemporaneous or later information; the workflow is suited to retrospective analysis and scenario-based prediction, not strict pre-event forecasting without further changes.

A local rerun of the two available category workbooks with the original selected-model code and current dependencies (scikit-learn 1.8.0) returned accuracy/macro F1 of 0.6207/0.5926 for feminine hygiene and 0.7576/0.7299 for sunscreen. The original course scores are retained in the results table; the cause of the differences has not been established.

## Exploratory Revenue Proxy

The original two-stage design uses the predicted recovery class as an additional feature in XGBoost regression. The intended business quantity is promotion-period revenue minus estimated baseline revenue over promotion duration plus recovery days. In the available event tables, a baseline daily revenue field is absent, so the code uses a fallback proxy based on baseline quantity, discount depth and first-day demand. This is not a measured counterfactual revenue or realized incremental profit. The regression is trained with observed K classes and evaluated with predicted classes, making its metrics exploratory rather than a deployment-ready economic estimate.
