import pandas as pd
import numpy as np
import os, warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from features.feature_engineering import engineer_features, get_feature_columns, get_targets

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved')
os.makedirs(MODELS_DIR, exist_ok=True)


def train_models(data_path=None, df=None, n_folds=5):
    if df is None:
        df = pd.read_csv(data_path)

    print("Engineering features...")
    df = engineer_features(df)

    features = get_feature_columns()
    targets  = get_targets()
    X = df[features].fillna(df[features].median())

    all_results    = {}
    cv_results_rows = []

    # Only XGBoost + RandomForest (Neural Net dropped — poor AUC on imbalanced data)
    model_configs = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_split=5, random_state=42, n_jobs=-1
        ),
    }

    for target in targets:
        print(f"\n{'='*55}")
        print(f"  Target: {target}")
        print(f"{'='*55}")

        y   = df[target]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        target_results = {}

        for name, model in model_configs.items():
            print(f"\n  {name}")

            # Cross-validation
            cv = cross_validate(
                model, X, y, cv=skf,
                scoring=['accuracy','f1_weighted','roc_auc'],
                return_train_score=True, n_jobs=-1
            )

            train_acc = cv['train_accuracy'].mean()
            cv_acc    = cv['test_accuracy'].mean()
            cv_acc_sd = cv['test_accuracy'].std()
            cv_f1     = cv['test_f1_weighted'].mean()
            cv_auc    = cv['test_roc_auc'].mean()
            overfit   = train_acc - cv_acc

            print(f"    Train Accuracy : {train_acc:.4f}")
            print(f"    CV Accuracy    : {cv_acc:.4f} ± {cv_acc_sd:.4f}  "
                  f"{'⚠  OVERFIT' if overfit > 0.15 else '✓ OK'}")
            print(f"    CV F1 Score    : {cv_f1:.4f}")
            print(f"    CV ROC-AUC     : {cv_auc:.4f}")
            print(f"    Overfit Gap    : {overfit:.4f}")

            # Retrain on full data
            model.fit(X, y)
            joblib.dump(model, os.path.join(MODELS_DIR, f'{name}_{target}.pkl'))

            target_results[name] = {
                'model'     : model,
                'train_acc' : round(train_acc, 4),
                'cv_acc'    : round(cv_acc,    4),
                'cv_acc_sd' : round(cv_acc_sd, 4),
                'cv_f1'     : round(cv_f1,     4),
                'cv_auc'    : round(cv_auc,    4),
                'overfit'   : round(overfit,   4),
            }

            cv_results_rows.append({
                'Target'   : target,
                'Model'    : name,
                'Train_Acc': round(train_acc, 4),
                'Accuracy' : round(cv_acc,    4),
                'Acc_Std'  : round(cv_acc_sd, 4),
                'F1_Score' : round(cv_f1,     4),
                'ROC_AUC'  : round(cv_auc,    4),
                'Overfit'  : round(overfit,   4),
            })

        all_results[target] = target_results

    # Summary
    print(f"\n\n{'='*65}")
    print("  CROSS-VALIDATION SUMMARY  (5-fold · XGBoost + RandomForest)")
    print(f"{'='*65}")
    for target in targets:
        print(f"\n  Target: {target}")
        print(f"  {'Model':<15} {'TrainAcc':>9} {'CV Acc':>9} {'±':>6} "
              f"{'F1':>8} {'AUC':>8} {'Overfit':>9}")
        print(f"  {'-'*60}")
        for name, m in all_results[target].items():
            flag = ' ⚠' if m['overfit'] > 0.15 else ' ✓'
            print(f"  {name:<15} {m['train_acc']:>9.4f} {m['cv_acc']:>9.4f} "
                  f"{m['cv_acc_sd']:>6.4f} {m['cv_f1']:>8.4f} "
                  f"{m['cv_auc']:>8.4f} {m['overfit']:>8.4f}{flag}")

    cv_df = pd.DataFrame(cv_results_rows)
    cv_df.to_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'), index=False)
    print(f"\n  Saved → models/saved/model_comparison.csv")
    print(f"  NOTE: Use CV Accuracy for honest reporting, not Train Accuracy.")

    return all_results


if __name__ == '__main__':
    train_models(data_path=os.path.join(
        os.path.dirname(__file__), '..', 'data', 'f1_raw_data.csv'))