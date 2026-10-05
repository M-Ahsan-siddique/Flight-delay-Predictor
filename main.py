import os
import pandas as pd
from src.preprocessing import FlightPreprocessor
from src.models import ModelPipeline

def main():
    # 1. Define paths
    data_dir = 'dataset/First dataset/archive (2)'
    models_dir = 'models'
    reports_dir = 'reports/figures'
    
    print("="*60)
    print("Flight Delay Prediction ML Pipeline")
    print("="*60)
    
    # 2. Run Preprocessing
    # We sample 200,000 flights to ensure fast training and low memory usage
    preprocessor = FlightPreprocessor(sample_size=200000, random_state=42)
    X_train, y_train, X_test, y_test, train_df, test_df = preprocessor.run_pipeline(data_dir)
    
    print(f"\nFinal preprocessing shape:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 3. Model Training & Comparison
    pipeline = ModelPipeline(random_state=42)
    comparison_df = pipeline.train_and_evaluate_all(X_train, y_train, X_test, y_test, reports_dir)
    
    print("\n" + "="*50)
    print("Model Comparison Metrics:")
    print("="*50)
    print(comparison_df.to_string())
    print("="*50)
    
    # Save comparison metrics as CSV
    metrics_path = os.path.join('reports', 'model_comparison_metrics.csv')
    comparison_df.to_csv(metrics_path)
    print(f"Model comparison table saved to {metrics_path}")
    
    # Plot and save confusion matrices
    pipeline.plot_confusion_matrices(X_test, y_test, reports_dir)
    
    # 4. Hyperparameter Tuning
    # Identify the best performing model based on ROC-AUC
    best_model_name = comparison_df['ROC-AUC'].idxmax()
    print(f"\nBest performing model: {best_model_name}")
    
    tuned_model = pipeline.tune_best_model(X_train, y_train, X_test, y_test, best_model_name)
    
    # 5. Save the final pipeline
    pipeline.save_pipeline(tuned_model, preprocessor, models_dir)
    print("\nEnd-to-End Flight Delay Prediction Pipeline Run Complete!")
    print("="*60)

if __name__ == '__main__':
    main()
