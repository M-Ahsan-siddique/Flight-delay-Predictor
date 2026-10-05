import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

class ModelPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=random_state),
            'HistGradientBoosting': HistGradientBoostingClassifier(random_state=random_state)
        }
        self.trained_models = {}
        self.comparison_metrics = {}

    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test, reports_dir):
        """
        Trains all models, computes evaluation metrics, and plots comparisons.
        """
        print("Starting model training and evaluation...")
        os.makedirs(reports_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)
            
            self.comparison_metrics[name] = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC-AUC': auc
            }
            
            # Plot ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
            
            print(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {auc:.4f}")
            
        # Finish ROC Curve plot
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        roc_path = os.path.join(reports_dir, 'roc_curves.png')
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"ROC Curves comparison plot saved to {roc_path}")
        
        comparison_df = pd.DataFrame(self.comparison_metrics).T
        return comparison_df

    def plot_confusion_matrices(self, X_test, y_test, reports_dir):
        """
        Plots confusion matrices for all trained models.
        """
        n_models = len(self.trained_models)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(self.trained_models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
            axes[idx].set_title(f'Confusion Matrix - {name}')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xticklabels(['On-Time', 'Delayed'])
            axes[idx].set_yticklabels(['On-Time', 'Delayed'])
            
        plt.tight_layout()
        cm_path = os.path.join(reports_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Confusion matrices plot saved to {cm_path}")

    def tune_best_model(self, X_train, y_train, X_test, y_test, best_model_name):
        """
        Performs RandomizedSearchCV on the best performing model to optimize parameters.
        """
        print(f"\nStarting hyperparameter tuning for {best_model_name}...")
        
        if best_model_name == 'HistGradientBoosting':
            model = HistGradientBoostingClassifier(random_state=self.random_state)
            param_dist = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_iter': [50, 100, 150],
                'max_depth': [3, 5, 8, 12, None],
                'l2_regularization': [0.0, 0.1, 1.0, 10.0]
            }
        elif best_model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            param_dist = {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            print(f"Tuning not configured for {best_model_name}. Returning baseline model.")
            return self.trained_models[best_model_name]

        # Use 3-fold CV and 10 iterations to run fast
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=8,
            scoring='roc_auc',
            cv=3,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        print("Best parameters found:", search.best_params_)
        
        best_model = search.best_estimator_
        
        # Evaluate tuned model
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"Tuned {best_model_name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {auc:.4f}")
        return best_model

    def save_pipeline(self, model, preprocessor, models_dir):
        """
        Saves the final model and preprocessor to disk.
        """
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(models_dir, 'final_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Save preprocessor
        prep_path = os.path.join(models_dir, 'preprocessor.pkl')
        with open(prep_path, 'wb') as f:
            pickle.dump(preprocessor, f)
            
        print(f"Successfully saved final model and preprocessor to {models_dir}")
