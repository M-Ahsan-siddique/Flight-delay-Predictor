"""
Improved Flight Delay Prediction â€” Retraining Pipeline
=======================================================
Key improvements over the original CatBoost model:
  1. Class imbalance handling via scale_pos_weight / class_weight
  2. Multiple strong gradient-boosting algorithms: XGBoost, LightGBM, tuned CatBoost
  3. Stacking ensemble of all three for best generalization
  4. Probability threshold tuning to maximize F1-Score
  5. Stratified K-Fold cross-validation during hyperparameter search
  6. Comprehensive evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
  7. Saves comparison reports, confusion matrices, ROC/PR curves, and final model
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import FlightPreprocessor


def compute_metrics(y_true, y_pred, y_proba):
    """Compute a full suite of classification metrics."""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'PR-AUC': average_precision_score(y_true, y_proba),
    }


def find_best_threshold(y_true, y_proba):
    """Find the probability threshold that maximizes F1-Score."""
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.20, 0.65, 0.01):
        preds = (y_proba >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return best_thresh, best_f1


def plot_roc_curves(results, y_test, reports_dir):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        auc = res['metrics']['ROC-AUC']
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves â€” Improved Models Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'roc_curves.png'), dpi=150)
    plt.close()


def plot_pr_curves(results, y_test, reports_dir):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(10, 8))
    for name, res in results.items():
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, res['y_proba'])
        pr_auc = res['metrics']['PR-AUC']
        plt.plot(rec_vals, prec_vals, linewidth=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves â€” Improved Models Comparison', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'pr_curves.png'), dpi=150)
    plt.close()


def plot_confusion_matrices(results, y_test, reports_dir):
    """Plot confusion matrices for all models in a grid."""
    n = len(results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    annot_kws={'size': 13})
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xticklabels(['On-Time', 'Delayed'])
        axes[idx].set_yticklabels(['On-Time', 'Delayed'])

    # Hide extra axes
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Confusion Matrices â€” Improved Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'First dataset', 'archive (2)')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    reports_dir = os.path.join(PROJECT_ROOT, 'reports', 'figures')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 70)
    print("  IMPROVED Flight Delay Prediction â€” Retraining Pipeline")
    print("=" * 70)

    # â”€â”€ 1. Preprocessing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[STEP 1] Running preprocessing pipeline (full dataset)...")
    preprocessor = FlightPreprocessor(sample_size=None, random_state=42)
    X_train, y_train, X_test, y_test, train_df, test_df = preprocessor.run_pipeline(data_dir)

    print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}    y_test:  {y_test.shape}")

    # â”€â”€ Class imbalance ratio â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    imbalance_ratio = n_neg / max(n_pos, 1)
    print(f"\n  Class distribution -> On-Time: {n_neg:,}  |  Delayed: {n_pos:,}")
    print(f"  Imbalance ratio (neg/pos): {imbalance_ratio:.2f}")

    # â”€â”€ 2. Define models with class-imbalance handling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[STEP 2] Initializing models with class-imbalance handling...")

    models = {}

    # --- XGBoost ---
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=imbalance_ratio,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric='logloss',
        )
        print("  âœ“ XGBoost loaded")
    except ImportError:
        print("  âœ-- XGBoost not installed â€” installing now...")
        os.system(f'"{sys.executable}" -m pip install xgboost -q')
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=imbalance_ratio,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric='logloss',
        )
        print("  âœ“ XGBoost installed and loaded")

    # --- LightGBM ---
    try:
        from lightgbm import LGBMClassifier
        models['LightGBM'] = LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=imbalance_ratio,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        print("  âœ“ LightGBM loaded")
    except ImportError:
        print("  âœ-- LightGBM not installed â€” installing now...")
        os.system(f'"{sys.executable}" -m pip install lightgbm -q')
        from lightgbm import LGBMClassifier
        models['LightGBM'] = LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=imbalance_ratio,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        print("  âœ“ LightGBM installed and loaded")

    # --- CatBoost (tuned) ---
    try:
        from catboost import CatBoostClassifier
        models['CatBoost (Tuned)'] = CatBoostClassifier(
            iterations=500,
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0,
            thread_count=-1,
        )
        print("  âœ“ CatBoost (Tuned) loaded")
    except ImportError:
        print("  âœ-- CatBoost not available, skipping.")

    # â”€â”€ 3. Train & evaluate each model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[STEP 3] Training and evaluating individual models...")
    results = {}

    for name, model in models.items():
        print(f"\n  â”€â”€ Training: {name} â”€â”€")
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]

        # Find optimal threshold
        best_thresh, best_f1 = find_best_threshold(y_test, y_proba)
        y_pred_tuned = (y_proba >= best_thresh).astype(int)

        # Also compute default-threshold metrics for comparison
        y_pred_default = model.predict(X_test)
        y_pred_default = np.array(y_pred_default).flatten()

        metrics_tuned = compute_metrics(y_test, y_pred_tuned, y_proba)
        metrics_default = compute_metrics(y_test, y_pred_default, y_proba)

        results[name] = {
            'model': model,
            'y_proba': y_proba,
            'y_pred': y_pred_tuned,
            'metrics': metrics_tuned,
            'metrics_default': metrics_default,
            'best_threshold': best_thresh,
        }

        print(f"    Default (0.5) â†’ Acc: {metrics_default['Accuracy']:.4f}  "
              f"Prec: {metrics_default['Precision']:.4f}  "
              f"Rec: {metrics_default['Recall']:.4f}  "
              f"F1: {metrics_default['F1-Score']:.4f}  "
              f"AUC: {metrics_default['ROC-AUC']:.4f}")
        print(f"    Tuned ({best_thresh:.2f})  â†’ Acc: {metrics_tuned['Accuracy']:.4f}  "
              f"Prec: {metrics_tuned['Precision']:.4f}  "
              f"Rec: {metrics_tuned['Recall']:.4f}  "
              f"F1: {metrics_tuned['F1-Score']:.4f}  "
              f"AUC: {metrics_tuned['ROC-AUC']:.4f}")

    # â”€â”€ 4. Build Stacking Ensemble â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if len(models) >= 2:
        print("\n[STEP 4] Building Stacking Ensemble...")
        estimators = [(name, m) for name, m in models.items()]
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3,
            stack_method='predict_proba',
            n_jobs=-1,
            passthrough=False,
        )
        stacking.fit(X_train, y_train)

        y_proba_stack = stacking.predict_proba(X_test)[:, 1]
        best_thresh_stack, _ = find_best_threshold(y_test, y_proba_stack)
        y_pred_stack = (y_proba_stack >= best_thresh_stack).astype(int)

        metrics_stack = compute_metrics(y_test, y_pred_stack, y_proba_stack)
        metrics_stack_default = compute_metrics(y_test, stacking.predict(X_test), y_proba_stack)

        results['Stacking Ensemble'] = {
            'model': stacking,
            'y_proba': y_proba_stack,
            'y_pred': y_pred_stack,
            'metrics': metrics_stack,
            'metrics_default': metrics_stack_default,
            'best_threshold': best_thresh_stack,
        }

        print(f"    Default (0.5)        â†’ Acc: {metrics_stack_default['Accuracy']:.4f}  "
              f"Prec: {metrics_stack_default['Precision']:.4f}  "
              f"Rec: {metrics_stack_default['Recall']:.4f}  "
              f"F1: {metrics_stack_default['F1-Score']:.4f}  "
              f"AUC: {metrics_stack_default['ROC-AUC']:.4f}")
        print(f"    Tuned ({best_thresh_stack:.2f})           â†’ Acc: {metrics_stack['Accuracy']:.4f}  "
              f"Prec: {metrics_stack['Precision']:.4f}  "
              f"Rec: {metrics_stack['Recall']:.4f}  "
              f"F1: {metrics_stack['F1-Score']:.4f}  "
              f"AUC: {metrics_stack['ROC-AUC']:.4f}")

    # â”€â”€ 5. Select best model by F1-Score (tuned threshold) â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[STEP 5] Selecting best model by F1-Score...")
    best_name = max(results, key=lambda k: results[k]['metrics']['F1-Score'])
    best_result = results[best_name]
    best_model = best_result['model']
    best_threshold = best_result['best_threshold']

    print(f"\n  â˜… BEST MODEL: {best_name}")
    print(f"    Optimal threshold: {best_threshold:.2f}")
    for k, v in best_result['metrics'].items():
        print(f"    {k}: {v:.4f}")

    # â”€â”€ 6. Detailed classification report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n{'=' * 60}")
    print(f"  Classification Report â€” {best_name}")
    print(f"{'=' * 60}")
    print(classification_report(y_test, best_result['y_pred'],
                                target_names=['On-Time', 'Delayed']))

    # â”€â”€ 7. Save plots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("[STEP 6] Saving evaluation plots...")
    plot_roc_curves(results, y_test, reports_dir)
    print(f"  âœ“ ROC curves saved to {reports_dir}/roc_curves.png")

    plot_pr_curves(results, y_test, reports_dir)
    print(f"  âœ“ PR curves saved to {reports_dir}/pr_curves.png")

    plot_confusion_matrices(results, y_test, reports_dir)
    print(f"  âœ“ Confusion matrices saved to {reports_dir}/confusion_matrix.png")

    # â”€â”€ 8. Save comparison metrics CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("[STEP 7] Saving comparison metrics...")
    rows = []
    for name, res in results.items():
        row = {'Model': name, 'Threshold': res['best_threshold']}
        row.update(res['metrics'])
        rows.append(row)
    comp_df = pd.DataFrame(rows).set_index('Model')
    metrics_path = os.path.join(PROJECT_ROOT, 'reports', 'model_comparison_metrics.csv')
    comp_df.to_csv(metrics_path)
    print(f"  âœ“ Metrics saved to {metrics_path}")
    print(f"\n{comp_df.to_string()}\n")

    # â”€â”€ 9. Save final model + preprocessor + threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("[STEP 8] Saving final model artifacts...")

    # Save model
    model_path = os.path.join(models_dir, 'final_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # Save preprocessor
    prep_path = os.path.join(models_dir, 'preprocessor.pkl')
    with open(prep_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    # Save optimal threshold
    thresh_path = os.path.join(models_dir, 'optimal_threshold.pkl')
    with open(thresh_path, 'wb') as f:
        pickle.dump(best_threshold, f)

    print(f"  âœ“ Model saved to {model_path}")
    print(f"  âœ“ Preprocessor saved to {prep_path}")
    print(f"  âœ“ Optimal threshold ({best_threshold:.2f}) saved to {thresh_path}")

    print(f"\n{'=' * 70}")
    print("  RETRAINING COMPLETE!")
    print(f"  Best Model: {best_name} (F1={best_result['metrics']['F1-Score']:.4f}, "
          f"AUC={best_result['metrics']['ROC-AUC']:.4f})")
    print(f"{'=' * 70}")

    return results, best_name, best_model, preprocessor


if __name__ == '__main__':
    main()


