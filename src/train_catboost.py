import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

# Add the project directory to python path
sys.path.append(os.path.abspath('.'))
from src.preprocessing import FlightPreprocessor

def main():
    data_dir = 'dataset/First dataset/archive (2)'
    models_dir = 'models'
    reports_dir = 'reports/figures'
    
    print("="*60)
    print("Training CatBoost Classifier on FULL DATASET (~5.23M Flights)")
    print("="*60)
    
    # 1. Run Preprocessing on full dataset
    preprocessor = FlightPreprocessor(sample_size=None, random_state=42)
    X_train, y_train, X_test, y_test, train_df, test_df = preprocessor.run_pipeline(data_dir)
    
    print(f"\nFinal preprocessing shape for CatBoost model:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 2. Train CatBoost Classifier
    # thread_count=-1 enables multi-threaded training on all CPU cores
    print("\nTraining CatBoost Classifier on full training set...")
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=20,
        thread_count=-1
    )
    
    model.fit(X_train, y_train)
    
    # 3. Predict and Evaluate
    print("\nEvaluating model on full test set...")
    y_pred = model.predict(X_test)
    # CatBoost returns shape (N,) or (N,1) for predict. Let's make it flat.
    y_pred = np.array(y_pred).flatten()
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    print("\n" + "="*50)
    print("CATBOOST CLASSIFICATION METRICS:")
    print("="*50)
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"Precision:        {prec * 100:.2f}%")
    print(f"Recall:           {rec * 100:.2f}%")
    print(f"F1-Score:         {f1 * 100:.2f}%")
    print(f"ROC-AUC:          {auc:.4f}")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # 4. Plot and Save Visualizations
    os.makedirs(reports_dir, exist_ok=True)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - CatBoost Classifier (Full)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['On-Time', 'Delayed'])
    plt.yticks([0.5, 1.5], ['On-Time', 'Delayed'])
    plt.tight_layout()
    cm_path = os.path.join(reports_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    # Plot ROC Curve
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"CatBoost Classifier (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - CatBoost Model')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(reports_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=150)
    plt.close()
    
    # 5. Save comparison metrics as CSV
    comparison_df = pd.DataFrame([{
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    }], index=['CatBoost (Full Dataset)'])
    metrics_path = os.path.join('reports', 'model_comparison_metrics.csv')
    comparison_df.to_csv(metrics_path)
    
    # 6. Save the final pipeline
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, 'final_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    # Save preprocessor
    prep_path = os.path.join(models_dir, 'preprocessor.pkl')
    with open(prep_path, 'wb') as f:
        pickle.dump(preprocessor, f)
        
    print(f"\nSaved final CatBoost model and preprocessor to {models_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
