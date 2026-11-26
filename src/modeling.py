"""Category-specific classification and exploratory revenue-proxy regression."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier, XGBRegressor
from .model_config import CONFIGS

TARGET = 'K_consecutive_2days'
# Preserve the original LabelEncoder ordering (Medium, Low, High).
LABELS = ['Low', 'Medium', 'High']
CLASS_ORDER = ['Medium', 'Low', 'High']
LOG_FEATURES = ['promo_period_first_demand', 'Baseline_14d_qty']


def prepare(df, features, category, *, drop_complete=False, log=True):
    """Match the historical preprocessing and category thresholds."""
    frame = df[list(dict.fromkeys(features + [TARGET]))].copy()
    if drop_complete:
        frame = frame.dropna().copy()
    else:
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET]).copy()
        for col in features:
            if frame[col].isna().any():
                median = frame[col].median()
                frame[col] = frame[col].fillna(median if pd.notna(median) else 0)
    labels = pd.cut(frame[TARGET], bins=CONFIGS[category]['bins'] + [np.inf], labels=LABELS)
    if labels.isna().any():
        raise ValueError('Recovery labels require positive observed K values.')
    y = labels.map({label: i for i, label in enumerate(CLASS_ORDER)}).astype(int).to_numpy()
    if log:
        for col in LOG_FEATURES:
            if col in frame:
                frame[col] = np.log1p(frame[col])
    return frame, y


def split_scaled(frame, y, features, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(frame[features], y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler()
    return X_train, X_test, y_train, y_test, scaler.fit_transform(X_train), scaler.transform(X_test)


def fit_model(key, X_train, y_train, *, baseline=False):
    classes = np.unique(y_train)
    weights = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y_train)))
    models = {
        'LR': LogisticRegression(class_weight=weights, max_iter=200, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, class_weight=weights, probability=baseline, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'RF': RandomForestClassifier(n_estimators=100, class_weight=weights, random_state=42),
        'XGB': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='mlogloss'),
    }
    model = models[key]
    if key == 'XGB':
        model.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))
    else:
        model.fit(X_train, y_train)
    return model


def score_model(key, frame, y, features, baseline=False):
    _, _, yt, yv, xt, xv = split_scaled(frame, y, features)
    pred = fit_model(key, xt, yt, baseline=baseline).predict(xv)
    return accuracy_score(yv, pred), f1_score(yv, pred, average='macro')


def compare_models(df, category):
    cfg = CONFIGS[category]
    features = cfg['baseline_features']
    frame, y = prepare(df, features, category, drop_complete=True, log=cfg['baseline_log'])
    rows = []
    for key in ['LR', 'SVM', 'KNN', 'RF', 'XGB']:
        acc, f1 = score_model(key, frame, y, features, baseline=True)
        rows.append({'model': key, 'accuracy': acc, 'macro_f1': f1})
    return pd.DataFrame(rows)


def forward_selection(df, category):
    """Historical one-pass forward selection on a fixed stratified split."""
    cfg = CONFIGS[category]
    start = cfg['forward_start']
    candidates = [f for f in cfg['forward_candidates'] if f in df and f not in start]
    frame, y = prepare(df, start + candidates, category)
    rows = []
    for key in ['LR', 'SVM', 'KNN', 'RF', 'XGB']:
        chosen = start.copy()
        acc, best = score_model(key, frame, y, chosen)
        initial = best
        for feature in candidates:
            a, f = score_model(key, frame, y, chosen + [feature])
            if f > best:
                chosen.append(feature)
                acc, best = a, f
        rows.append({'model': key, 'initial_macro_f1': initial, 'macro_f1': best, 'accuracy': acc, 'features': chosen})
    return pd.DataFrame(rows)


def final_knn(df, category):
    features = CONFIGS[category]['final_features']
    frame, y = prepare(df, features, category)
    _, _, yt, yv, xt, xv = split_scaled(frame, y, features)
    model = fit_model('KNN', xt, yt)
    pred = model.predict(xv)
    metrics = {'accuracy': accuracy_score(yv, pred), 'macro_f1': f1_score(yv, pred, average='macro'), 'test_events': len(yv)}
    cm = confusion_matrix(yv, pred, labels=[0, 1, 2])
    importance = permutation_importance(model, xv, yv, scoring='f1_macro', n_repeats=5, random_state=42)
    diagnostics = pd.DataFrame({'feature': features, 'permutation_importance': importance.importances_mean,
                                'spearman_with_K': [frame[c].corr(frame[TARGET], method='spearman') for c in features]})
    return metrics, cm, diagnostics.sort_values('permutation_importance', ascending=False)


def exploratory_revenue_proxy(df, category):
    """Retain the historical two-stage design; return aggregate metrics only.

    The fallback baseline is a proxy, not a measured counterfactual revenue.
    """
    cfg = CONFIGS[category]
    features = cfg['regression_stage1_features']
    frame, y = prepare(df, features, category)
    x_train, x_test, yt, yv, xt, xv = split_scaled(frame, y, features, cfg['regression_seed'])
    classifier = fit_model('KNN', xt, yt)
    pred = classifier.predict(xv)
    k = df[TARGET]
    revenue = df['promo_total_sales_amount'] if 'promo_total_sales_amount' in df else (df['promo_days'] + k) * df['avg_day_SubtotalPDPrice']
    if 'avg_day_baseline_sales_amount' in df:
        baseline = (df['promo_days'] + k) * df['avg_day_baseline_sales_amount']
        target_definition = 'baseline revenue estimate'
    else:
        baseline = df['Baseline_14d_qty'] * (df['avg_day_SubtotalPDPrice'] / (df['promo_period_first_demand'] + 1))
        target_definition = 'historical fallback proxy'
    target = revenue - baseline
    target = target.fillna(target.median())
    base = cfg['regression_features']
    train = x_train[base].copy()
    test = x_test[base].copy()
    # Actual K class for training, predicted K class for evaluation.
    for label in LABELS:
        i = CLASS_ORDER.index(label)
        train[f'K_input_{label}'] = (yt == i).astype(int)
        test[f'K_input_{label}'] = (pred == i).astype(int)
    scaler = StandardScaler()
    model = XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42, eval_metric='rmse')
    y_train = target.loc[x_train.index].to_numpy()
    y_test = target.loc[x_test.index].to_numpy()
    model.fit(scaler.fit_transform(train), y_train)
    pred_revenue = model.predict(scaler.transform(test))
    return {'target_definition': target_definition, 'r2': r2_score(y_test, pred_revenue),
            'mae': mean_absolute_error(y_test, pred_revenue), 'rmse': np.sqrt(mean_squared_error(y_test, pred_revenue))}
