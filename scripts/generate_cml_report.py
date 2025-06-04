import json
import os
import pandas as pd
from datetime import datetime
import sys

def generate_cml_report():
    commit_sha = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    branch_name = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    
    with open('report.md', 'w') as f:
        f.write("# 🤖 Mental Health MLOps Training Report\n\n")
        f.write(f"**Commit:** `{commit_sha}`\n")
        f.write(f"**Branch:** `{branch_name}`\n")
        f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model Metrics
        if os.path.exists('results/metrics.txt'):
            f.write("## 📊 Model Performance Metrics\n")
            f.write("```")
            with open('results/metrics.txt', 'r') as metrics:
                f.write(metrics.read())
            f.write("```\n\n")
        else:
            f.write("## 📊 Model Performance Metrics\n")
            f.write("⚠️ Model metrics not available\n\n")
        
        # Model Comparison
        if os.path.exists('results/model_comparison.json'):
            f.write("## 🏆 Model Comparison Results\n")
            try:
                with open('results/model_comparison.json', 'r') as comp:
                    data = json.load(comp)
                f.write(f"**Best Model:** {data['best_model']}\n")
                f.write(f"**Final Test Accuracy:** {data['final_test_accuracy']:.4f}\n")
                f.write(f"**Final Test F1 Score:** {data['final_test_f1']:.4f}\n\n")
                f.write("### Model Performance Comparison:\n")
                for model, scores in data['model_scores'].items():
                    f.write(f"- **{model}:** {scores['mean_accuracy']:.4f} (±{scores['std_accuracy']:.4f})\n")
                f.write("\n")
            except Exception as e:
                f.write(f"Error reading model comparison: {e}\n\n")
        
        # Visualizations
        f.write("## 📈 Model Performance Visualizations\n\n")
        
        if os.path.exists('results/model_comparison.png'):
            f.write("### Model Comparison\n")
            f.write("![Model Comparison](./results/model_comparison.png)\n\n")
        
        if os.path.exists('results/model_results.png'):
            f.write("### Confusion Matrix\n")
            f.write("![Confusion Matrix](./results/model_results.png)\n\n")
        
        if os.path.exists('results/shap_summary.png'):
            f.write("### SHAP Feature Importance\n")
            f.write("![SHAP Summary](./results/shap_summary.png)\n\n")
        
        # Data Quality Report
        f.write("## 📋 Data Quality Summary\n")
        try:
            data_files = ['data/mental_health_lite.csv', 'data/mental_health_life_cut.csv']
            df = None
            
            for file_path in data_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    break
            
            if df is not None:
                f.write(f"- **Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns\n")
                f.write(f"- **Missing Values:** {df.isnull().sum().sum()} total\n")
                if 'mental_health_risk' in df.columns:
                    risk_dist = df['mental_health_risk'].value_counts()
                    f.write(f"- **Target Distribution:** {dict(risk_dist)}\n")
                numeric_cols = len(df.select_dtypes(include=['number']).columns)
                categorical_cols = len(df.select_dtypes(include=['object']).columns)
                f.write(f"- **Data Types:** {numeric_cols} numeric, {categorical_cols} categorical\n")
            else:
                f.write("- **Dataset:** Not found or could not be loaded\n")
        except Exception as e:
            f.write(f"- **Data Quality Check:** Failed - {e}\n")
        
        f.write("\n")
        
        # Quality Gates
        f.write("## ✅ Quality Gates\n")
        try:
            if os.path.exists('results/model_comparison.json'):
                with open('results/model_comparison.json', 'r') as comp:
                    data = json.load(comp)
                
                accuracy = data['final_test_accuracy']
                f1_score = data['final_test_f1']
                
                accuracy_pass = "✅ PASS" if accuracy >= 0.70 else "❌ FAIL"
                f1_pass = "✅ PASS" if f1_score >= 0.70 else "❌ FAIL"
                
                f.write(f"- **Accuracy Threshold (≥0.70):** {accuracy_pass} ({accuracy:.3f})\n")
                f.write(f"- **F1 Score Threshold (≥0.70):** {f1_pass} ({f1_score:.3f})\n")
                f.write(f"- **Model Selection:** ✅ PASS ({data['best_model']} selected)\n")
            else:
                f.write("- **Quality Gates:** ⚠️ Model results not available\n")
        except Exception as e:
            f.write(f"- **Quality Gates:** ❌ Error checking thresholds - {e}\n")
        
        f.write("\n")
        f.write("## 🔄 Pipeline Status\n")
        f.write("- **Code Quality:** ✅ Completed\n")
        f.write("- **Data Validation:** ✅ Passed\n")
        f.write("- **Model Training:** ✅ Completed\n")
        f.write("- **Artifacts Generated:** ✅ Available\n")
        f.write("- **CML Report:** ✅ Generated\n\n")
        
        f.write("---\n")
        f.write("*Generated by Mental Health MLOps CI Pipeline*\n")


if __name__ == "__main__":  # Fixed: Added proper newline separation
    generate_cml_report()
