import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import skops.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_suite import DataDriftSuite
from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric, DatasetCorrelationsMetric

def setup_directories():
    """Setup direktori yang diperlukan"""
    directories = ["models", "results", "monitoring", "monitoring/evidently_reports"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories setup completed")

def load_and_prepare_data():
    """Load dan prepare data dengan Evidently monitoring integration"""
    print("🔍 Loading and preparing data...")
    
    # Path data files yang konsisten dengan evidently_monitoring.py
    data_files = [
        "data/mental_health_lite.csv", 
        "data/mental_health_life_cut.csv"
    ]
    
    df = None
    data_source = None
    
    # Cari file data yang tersedia
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ Found data file: {file_path}")
            try:
                df = pd.read_csv(file_path)
                data_source = file_path
                print(f"✅ Data loaded successfully from {file_path}")
                print(f"📊 Data shape: {df.shape}")
                break
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue
    
    if df is None:
        raise FileNotFoundError("❌ No valid dataset found in data/ folder")
    
    # Create data loading summary untuk monitoring
    data_summary = {
        "data_loaded": True,
        "source_file": data_source,
        "shape": df.shape,
        "columns": list(df.columns),
        "timestamp": datetime.now().isoformat(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict()
    }
    
    # Save data summary
    with open("monitoring/data_loading_summary.json", "w") as f:
        json.dump(data_summary, f, indent=2, default=str)
    
    return df, data_source

def create_evidently_data_report(df, data_source):
    """Create Evidently data quality report"""
    print("📊 Creating Evidently data quality report...")
    
    try:
        # Create basic data quality report
        data_report = Report(metrics=[
            DatasetMissingValuesMetric(),
            DatasetCorrelationsMetric(),
        ])
        
        # Run report
        data_report.run(reference_data=None, current_data=df)
        
        # Save HTML report
        report_path = "monitoring/evidently_reports/data_quality_report.html"
        data_report.save_html(report_path)
        
        # Extract key metrics
        report_dict = data_report.as_dict()
        
        data_quality_summary = {
            "report_generated": True,
            "timestamp": datetime.now().isoformat(),
            "data_source": data_source,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values_count": df.isnull().sum().sum(),
            "report_path": report_path
        }
        
        # Save summary
        with open("monitoring/evidently_data_quality.json", "w") as f:
            json.dump(data_quality_summary, f, indent=2)
        
        print(f"✅ Evidently data quality report saved to {report_path}")
        
    except Exception as e:
        print(f"⚠️ Evidently data quality report failed: {e}")
        # Create fallback summary
        fallback_summary = {
            "report_generated": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "data_source": data_source,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values_count": df.isnull().sum().sum()
        }
        
        with open("monitoring/evidently_data_quality.json", "w") as f:
            json.dump(fallback_summary, f, indent=2)

def encode_categorical_features(df):
    """Encode categorical features dengan proper handling"""
    print("🔧 Encoding categorical features...")
    
    encoders = {}
    
    # Identifikasi kolom kategorik
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Exclude target column jika ada
    target_candidates = ['mental_health_condition', 'target', 'label', 'class']
    target_column = None
    
    for col in target_candidates:
        if col in df.columns:
            target_column = col
            if col in categorical_columns:
                categorical_columns.remove(col)
            break
    
    print(f"🎯 Target column identified: {target_column}")
    print(f"📝 Categorical columns to encode: {categorical_columns}")
    
    # Encode categorical features
    for col in categorical_columns:
        if col in df.columns:
            encoder = LabelEncoder()
            # Handle missing values
            df[col] = df[col].fillna('unknown')
            df[f"{col}_encoded"] = encoder.fit_transform(df[col])
            encoders[col] = encoder
            print(f"✅ Encoded {col} -> {col}_encoded")
    
    # Encode target column jika kategorik
    if target_column and df[target_column].dtype == 'object':
        target_encoder = LabelEncoder()
        df[f"{target_column}_encoded"] = target_encoder.fit_transform(df[target_column])
        encoders['target'] = target_encoder
        target_column = f"{target_column}_encoded"
        print(f"✅ Encoded target column: {target_column}")
    
    return df, encoders, target_column

def prepare_features(df, target_column):
    """Prepare features untuk training"""
    print("🎯 Preparing features for training...")
    
    # Exclude non-feature columns
    exclude_columns = [
        target_column,
        'id', 'index', 'timestamp', 'date'
    ]
    
    # Add original categorical columns to exclude
    categorical_originals = [col for col in df.columns if col.endswith('_encoded')]
    for encoded_col in categorical_originals:
        original_col = encoded_col.replace('_encoded', '')
        if original_col in df.columns:
            exclude_columns.append(original_col)
    
    # Select feature columns
    feature_columns = []
    for col in df.columns:
        if col not in exclude_columns and df[col].dtype in ['int64', 'float64']:
            feature_columns.append(col)
    
    print(f"📊 Selected features: {feature_columns}")
    print(f"🎯 Target column: {target_column}")
    
    if len(feature_columns) == 0:
        raise ValueError("❌ No valid numeric features found for training")
    
    return feature_columns

def create_model_comparison_chart(models_results):
    """Create model performance comparison chart"""
    print("📊 Creating model comparison chart...")
    
    # Extract metrics
    model_names = []
    accuracies = []
    f1_scores = []
    
    for name, results in models_results.items():
        if 'error' not in results:
            model_names.append(name)
            accuracies.append(results['accuracy'])
            f1_scores.append(results['f1_weighted'])
    
    if not model_names:
        print("⚠️ No valid models to compare")
        return
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.annotate(f'{acc:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # F1 Score comparison
    bars2 = ax2.bar(model_names, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1-Score (Weighted)', fontsize=12)
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.annotate(f'{f1:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Model comparison chart saved to results/model_comparison.png")

def create_confusion_matrix_plot(y_test, y_pred, model_name):
    """Create confusion matrix visualization"""
    print(f"📊 Creating confusion matrix for {model_name}...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Confusion matrix saved to results/confusion_matrix.png")

def create_roc_curve_plot(model, X_test, y_test, model_name):
    """Create ROC curve for binary classification"""
    print(f"📊 Creating ROC curve for {model_name}...")
    
    try:
        # Check if binary classification
        if len(np.unique(y_test)) != 2:
            print("⚠️ ROC curve only supported for binary classification")
            return
            
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ ROC curve saved to results/roc_curve.png")
        
    except Exception as e:
        print(f"⚠️ Could not create ROC curve: {e}")

def create_feature_importance_plot(model, feature_columns, model_name):
    """Create feature importance visualization"""
    print(f"📊 Creating feature importance plot for {model_name}...")
    
    try:
        # Get feature importance dari model
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        elif hasattr(model.named_steps['classifier'], 'coef_'):
            importances = np.abs(model.named_steps['classifier'].coef_[0])
        else:
            print("⚠️ Model does not support feature importance extraction")
            return
        
        # Create feature importance dataframe
        feature_imp_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 15 features
        top_features = feature_imp_df.head(15)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title(f'Top 15 Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Feature importance plot saved to results/feature_importance.png")
        
    except Exception as e:
        print(f"⚠️ Could not create feature importance plot: {e}")

def create_training_metrics_summary_plot(models_results):
    """Create comprehensive training metrics summary"""
    print("📊 Creating training metrics summary plot...")
    
    metrics_data = []
    
    for name, results in models_results.items():
        if 'error' not in results:
            metrics_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'F1_Weighted': results['f1_weighted'],
                'F1_Macro': results['f1_macro'],
                'CV_Mean': results['cv_mean']
            })
    
    if not metrics_data:
        print("⚠️ No valid metrics data to plot")
        return
        
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Metrics Summary', fontsize=16, fontweight='bold')
    
    # Plot each metric
    metrics = ['Accuracy', 'F1_Weighted', 'F1_Macro', 'CV_Mean']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[i//2, i%2]
        bars = ax.bar(df_metrics['Model'], df_metrics[metric], color=color, alpha=0.8)
        ax.set_title(f'{metric.replace("_", " ")}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, df_metrics[metric]):
            height = bar.get_height()
            ax.annotate(f'{value:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/training_metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Training metrics summary saved to results/training_metrics_summary.png")

def create_model_visualizations(models_results, y_test, X_test, trained_models, feature_columns):
    """Create comprehensive model performance visualizations"""
    print("📊 Creating model performance visualizations...")
    
    # Setup plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model Comparison Chart
    create_model_comparison_chart(models_results)
    
    # 2. Training metrics summary
    create_training_metrics_summary_plot(models_results)
    
    # Find best model
    valid_models = [name for name in models_results.keys() if 'error' not in models_results[name]]
    if not valid_models:
        print("⚠️ No valid models found for detailed visualizations")
        return
        
    best_model_name = max(valid_models, key=lambda x: models_results[x]['accuracy'])
    
    if best_model_name in trained_models:
        best_model = trained_models[best_model_name]
        y_pred = best_model.predict(X_test)
        
        # 3. Confusion Matrix
        create_confusion_matrix_plot(y_test, y_pred, best_model_name)
        
        # 4. ROC Curve (jika binary classification)
        create_roc_curve_plot(best_model, X_test, y_test, best_model_name)
        
        # 5. Feature Importance
        create_feature_importance_plot(best_model, feature_columns, best_model_name)
    
    print("✅ Model visualizations created successfully")

def train_models(X_train, X_test, y_train, y_test, feature_columns):
    """Train multiple models dan return best model"""
    print("🤖 Training multiple models...")
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=6,
            learning_rate=0.1
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=6,
            learning_rate=0.1,
            verbose=-1
        )
    }
    
    # Create pipeline dengan StandardScaler
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        
        try:
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Cross validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            trained_models[name] = pipeline
            
            print(f"✅ {name} - Accuracy: {accuracy:.4f}, F1: {f1_weighted:.4f}")
            
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Find best model
    valid_models = [name for name in results.keys() if 'error' not in results[name]]
    if not valid_models:
        raise ValueError("❌ No models trained successfully")
        
    best_model_name = max(valid_models, key=lambda x: results[x]['accuracy'])
    best_model = trained_models[best_model_name]
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"🏆 Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    return best_model, best_model_name, results, trained_models

def save_model_and_metadata(model, model_name, results, feature_columns, encoders):
    """Save model dan metadata"""
    print("💾 Saving model and metadata...")
    
    # Save model dengan skops
    model_path = f"models/best_model_{model_name.lower()}.skops"
    sio.dump(model, model_path)
    
    # Save encoders
    encoders_path = "models/encoders.pkl"
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    
    # Create comprehensive metadata
    metadata = {
        'model_name': model_name,
        'model_path': model_path,
        'encoders_path': encoders_path,
        'feature_columns': feature_columns,
        'training_timestamp': datetime.now().isoformat(),
        'model_performance': results[model_name],
        'all_models_performance': results,
        'best_accuracy': results[model_name]['accuracy'],
        'model_type': 'classification',
        'framework': 'sklearn_pipeline'
    }
    
    # Save metadata
    metadata_path = "models/model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✅ Model saved to {model_path}")
    print(f"✅ Metadata saved to {metadata_path}")
    
    return metadata

def create_training_summary(metadata, data_source):
    """Create comprehensive training summary untuk monitoring"""
    print("📋 Creating training summary...")
    
    training_summary = {
        "training_completed": True,
        "timestamp": datetime.now().isoformat(),
        "data_source": data_source,
        "best_model": metadata['model_name'],
        "best_accuracy": metadata['best_accuracy'],
        "models_trained": list(metadata['all_models_performance'].keys()),
        "feature_count": len(metadata['feature_columns']),
        "status": "success",
        "model_path": metadata['model_path'],
        "metadata_path": "models/model_metadata.json",
        "visualizations_created": [
            "results/model_comparison.png",
            "results/confusion_matrix.png",
            "results/roc_curve.png",
            "results/feature_importance.png",
            "results/training_metrics_summary.png"
        ]
    }
    
    # Save training summary
    with open("monitoring/training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)
    
    # Save untuk CML report
    with open("results/training_results.json", "w") as f:
        json.dump(training_summary, f, indent=2)
    
    print("✅ Training summary created for monitoring integration")

def main():
    """Main training pipeline"""
    print("🚀 Starting ML Training Pipeline with Evidently Integration")
    print("=" * 60)
    
    try:
        # Setup directories
        setup_directories()
        
        # Load and prepare data
        df, data_source = load_and_prepare_data()
        
        # Create Evidently data quality report
        create_evidently_data_report(df, data_source)
        
        # Encode categorical features
        df, encoders, target_column = encode_categorical_features(df)
        
        if target_column is None:
            raise ValueError("❌ No target column found")
        
        # Prepare features
        feature_columns = prepare_features(df, target_column)
        
        # Prepare training data
        X = df[feature_columns]
        y = df[target_column]
        
        print(f"📊 Training data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Train models
        best_model, best_model_name, results, trained_models = train_models(
            X_train, X_test, y_train, y_test, feature_columns
        )
        
        # Create visualizations
        create_model_visualizations(results, y_test, X_test, trained_models, feature_columns)
        
        # Save model and metadata
        metadata = save_model_and_metadata(
            best_model, best_model_name, results, feature_columns, encoders
        )
        
        # Create training summary
        create_training_summary(metadata, data_source)
        
        print("=" * 60)
        print("🎉 Training pipeline completed successfully!")
        print(f"🏆 Best model: {best_model_name}")
        print(f"📊 Best accuracy: {metadata['best_accuracy']:.4f}")
        print("📊 Visualizations created:")
        print("   - Model comparison chart")
        print("   - Confusion matrix")
        print("   - ROC curve (if binary classification)")
        print("   - Feature importance plot")
        print("   - Training metrics summary")
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        
        # Create error summary
        error_summary = {
            "training_completed": False,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "failed"
        }
        
        with open("monitoring/training_summary.json", "w") as f:
            json.dump(error_summary, f, indent=2)
        
        with open("results/training_results.json", "w") as f:
            json.dump(error_summary, f, indent=2)
        
        raise

if __name__ == "__main__":
    main()