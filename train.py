import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import skops.io as sio
from evidently.presets import DataDriftPreset
from evidently import Report
from evidently import Dataset, DataDefinition
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_directories():
    """Setup direktori yang diperlukan"""
    directories = ["models", "results", "monitoring", "monitoring/evidently_reports"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories setup completed")

def load_and_prepare_data():
    """Load dan prepare data dengan proper encoding untuk menghindari serialization issues"""
    
    # Load data
    data_files = ['data/mental_health_lite.csv', 'data/mental_health_life_cut.csv']
    df = None
    
    for file_path in data_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded from: {file_path}")
            break
    
    if df is None:
        raise FileNotFoundError("No dataset found")
    
    print(f"📊 Original data shape: {df.shape}")
    print(f"📊 Original data types:\n{df.dtypes}")
    
    # ✅ CRITICAL FIX: Proper categorical encoding
    encoders = {}
    categorical_cols = ['gender', 'employment_status', 'work_environment', 
                       'mental_health_history', 'seeks_treatment']
    
    # Encode categorical variables PROPERLY
    for col in categorical_cols:
        if col in df.columns:
            print(f"🔧 Encoding {col}...")
            le = LabelEncoder()
            
            # Handle missing values first
            df[col] = df[col].fillna('Unknown')
            
            # Encode to numeric
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
            print(f"✅ {col} encoded: {le.classes_} -> {list(range(len(le.classes_)))}")
    
    # Encode target variable
    if 'mental_health_risk' in df.columns:
        le_risk = LabelEncoder()
        df['mental_health_risk'] = df['mental_health_risk'].fillna('Unknown')
        df['risk_encoded'] = le_risk.fit_transform(df['mental_health_risk'].astype(str))
        encoders['risk'] = le_risk
        print(f"✅ Target encoded: {le_risk.classes_} -> {list(range(len(le_risk.classes_)))}")
    
    # ✅ CRITICAL: Remove original categorical columns to avoid confusion
    columns_to_drop = [col for col in categorical_cols if col in df.columns]
    columns_to_drop.append('mental_health_risk')  # Remove original target
    
    df_clean = df.drop(columns=columns_to_drop, errors='ignore')
    
    # ✅ Ensure all remaining columns are numeric
    for col in df_clean.columns:
        if col != 'risk_encoded':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Fill any remaining NaN values
    df_clean = df_clean.fillna(0)
    
    print(f"📊 Cleaned data shape: {df_clean.shape}")
    print(f"📊 Cleaned data types:\n{df_clean.dtypes}")
    
    # Verify all data is numeric
    non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"⚠️ Non-numeric columns found: {non_numeric_cols}")
        for col in non_numeric_cols:
            if col != 'risk_encoded':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # Save encoders
    os.makedirs("model", exist_ok=True)
    with open("model/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    
    # Save reference data for monitoring
    reference_path = "monitoring/reference_data.csv"
    os.makedirs("monitoring", exist_ok=True)
    df_clean.to_csv(reference_path, index=False)
    print("✅ Reference data saved for future monitoring")
    
    return df_clean, encoders

def create_evidently_data_report(df, data_source):
    """Create Evidently data quality report"""
    print("📊 Creating Evidently data quality report...")
    
    try:
        from evidently import Report, Dataset, DataDefinition
        from evidently.presets import DataQualityPreset
        import os
        
        # Buat direktori jika belum ada
        os.makedirs("monitoring/evidently_reports", exist_ok=True)
        
        # Identifikasi tipe kolom secara otomatis
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Definisikan struktur data
        data_definition = DataDefinition(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )
        
        # Buat Dataset object
        current_data = Dataset.from_pandas(
            df,
            data_definition=data_definition
        )
        
        # Buat report dengan DataQualityPreset
        data_report = Report([
            DataQualityPreset()
        ])
        
        # Jalankan report
        data_report.run(current_data=current_data, reference_data=None)
        
        # Save HTML report
        report_path = "monitoring/evidently_reports/data_quality_report.html"
        data_report.save_html(report_path)
        
        # Extract key metrics dari report
        report_dict = data_report.as_dict()
        
        # Hitung statistik dasar
        missing_values_per_column = df.isnull().sum()
        total_missing = missing_values_per_column.sum()
        
        data_quality_summary = {
            "report_generated": True,
            "timestamp": datetime.now().isoformat(),
            "data_source": data_source,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numerical_columns": len(numerical_columns),
            "categorical_columns": len(categorical_columns),
            "missing_values_count": int(total_missing),
            "missing_values_percentage": round((total_missing / (len(df) * len(df.columns))) * 100, 2),
            "columns_with_missing": missing_values_per_column[missing_values_per_column > 0].to_dict(),
            "report_path": report_path,
            "data_definition": {
                "numerical_columns": numerical_columns,
                "categorical_columns": categorical_columns
            }
        }
        
        # Save summary
        with open("monitoring/evidently_data_quality.json", "w") as f:
            json.dump(data_quality_summary, f, indent=2)
        
        print(f"✅ Evidently data quality report saved to {report_path}")
        print(f"📈 Data summary: {len(df)} rows, {len(df.columns)} columns")
        print(f"🔢 Numerical columns: {len(numerical_columns)}")
        print(f"📝 Categorical columns: {len(categorical_columns)}")
        print(f"❌ Missing values: {total_missing} ({data_quality_summary['missing_values_percentage']}%)")
        
        return data_quality_summary
        
    except ImportError as e:
        print(f"⚠️ Import error - Evidently version mismatch: {e}")
        print("💡 Tip: Pastikan menggunakan evidently>=0.7.0")
        
        # Create fallback summary
        fallback_summary = {
            "report_generated": False,
            "error": f"Import error: {str(e)}",
            "error_type": "import_error",
            "timestamp": datetime.now().isoformat(),
            "data_source": data_source,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values_count": int(df.isnull().sum().sum())
        }
        
        with open("monitoring/evidently_data_quality.json", "w") as f:
            json.dump(fallback_summary, f, indent=2)
            
        return fallback_summary
        
    except Exception as e:
        print(f"⚠️ Evidently data quality report failed: {e}")
        
        # Create fallback summary
        fallback_summary = {
            "report_generated": False,
            "error": str(e),
            "error_type": "runtime_error",
            "timestamp": datetime.now().isoformat(),
            "data_source": data_source,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values_count": int(df.isnull().sum().sum())
        }
        
        with open("monitoring/evidently_data_quality.json", "w") as f:
            json.dump(fallback_summary, f, indent=2)
            
        return fallback_summary


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

def prepare_data_for_training(df, encoders):
    """Prepare data dengan proper data types untuk training"""
    print("🔧 Preparing data for training...")
    
    # Ensure we have a clean copy
    df_clean = df.copy()
    
    # Get feature columns (exclude target)
    feature_columns = [col for col in df_clean.columns if col != 'risk_encoded']
    
    print(f"📊 Available features: {len(feature_columns)}")
    print(f"Features: {feature_columns}")
    
    if len(feature_columns) == 0:
        raise ValueError("No feature columns found in dataset")
    
    # Prepare X and y
    X = df_clean[feature_columns].copy()
    y = df_clean['risk_encoded'].copy()
    
    # ✅ CRITICAL: Ensure all data is numeric and proper dtype
    print("🔧 Converting to proper numeric types...")
    
    # Convert X to float32 (consistent dtype)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(np.float32)
    
    # Convert y to int32
    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(np.int32)
    
    # Verify data types
    print(f"✅ X dtypes: {X.dtypes.unique()}")
    print(f"✅ y dtype: {y.dtype}")
    print(f"✅ Data prepared: X shape={X.shape}, y shape={y.shape}")
    
    # Final check - ensure no object dtypes remain
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        raise ValueError(f"Object columns still present: {object_cols}")
    
    return X, y, feature_columns

def train_models_fixed(X_train, X_test, y_train, y_test, feature_columns):
    """Train multiple models dengan improved error handling"""
    print("🤖 Training multiple models with fixed serialization...")
    
    # ✅ Ensure data is in correct format
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=50,  # Reduced for stability
            random_state=42,
            max_depth=8,
            min_samples_split=5,
            n_jobs=1  # ✅ Single job to avoid serialization issues
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=4,
            learning_rate=0.1,
            eval_metric='logloss',
            n_jobs=1  # ✅ Single job
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=4,
            learning_rate=0.1,
            verbose=-1,
            force_col_wise=True,
            n_jobs=1  # ✅ Single job
        )
    }
    
    results = {}
    trained_models = {}
    
    # ✅ Use simple cross-validation without parallelization
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            # ✅ Manual cross-validation to avoid serialization issues
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_fold = X_train.iloc[train_idx]
                X_val_fold = X_train.iloc[val_idx]
                y_train_fold = y_train.iloc[train_idx]
                y_val_fold = y_train.iloc[val_idx]
                
                # Clone model for each fold
                from sklearn.base import clone
                model_clone = clone(model)
                model_clone.fit(X_train_fold, y_train_fold)
                
                y_pred_fold = model_clone.predict(X_val_fold)
                score = accuracy_score(y_val_fold, y_pred_fold)
                cv_scores.append(score)
            
            cv_scores = np.array(cv_scores)
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Test predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Store results
            results[name] = {
                'accuracy': float(test_accuracy),  # ✅ Ensure serializable
                'f1_weighted': float(f1_weighted),
                'f1_macro': float(f1_macro),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'cv_scores': cv_scores.tolist()
            }
            
            trained_models[name] = model
            
            print(f"✅ {name} - CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f}), Test: {test_accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
            import traceback
            traceback.print_exc()
            
            results[name] = {
                'accuracy': 0.0,
                'f1_weighted': 0.0,
                'f1_macro': 0.0,
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'error': str(e)
            }
    
    # Filter out failed models
    valid_results = {name: result for name, result in results.items() if 'error' not in result}
    
    if not valid_results:
        print("⚠️ All models failed, creating dummy model...")
        # Create a simple dummy model that always works
        from sklearn.dummy import DummyClassifier
        dummy_model = DummyClassifier(strategy='most_frequent', random_state=42)
        dummy_model.fit(X_train, y_train)
        y_pred_dummy = dummy_model.predict(X_test)
        
        dummy_accuracy = accuracy_score(y_test, y_pred_dummy)
        
        results['DummyClassifier'] = {
            'accuracy': float(dummy_accuracy),
            'f1_weighted': float(f1_score(y_test, y_pred_dummy, average='weighted')),
            'f1_macro': float(f1_score(y_test, y_pred_dummy, average='macro')),
            'cv_mean': float(dummy_accuracy),
            'cv_std': 0.0,
            'cv_scores': [dummy_accuracy] * 3
        }
        
        trained_models['DummyClassifier'] = dummy_model
        valid_results = {'DummyClassifier': results['DummyClassifier']}
        print(f"✅ DummyClassifier - Accuracy: {dummy_accuracy:.4f}")
    
    # Select best model
    best_model_name = max(valid_results.keys(), key=lambda x: valid_results[x]['cv_mean'])
    best_model = trained_models[best_model_name]
    best_accuracy = valid_results[best_model_name]['accuracy']
    
    # Get predictions from best model
    y_pred_best = best_model.predict(X_test)
    
    print(f"🏆 Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    return best_model, best_model_name, results, y_pred_best

def create_feature_importance_plot(model, feature_columns, model_name):
    """Create dan save feature importance plot"""
    print("📊 Creating feature importance plot...")
    try:
        # Extract feature importance dari model
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        elif hasattr(model.named_steps['classifier'], 'coef_'):
            importances = np.abs(model.named_steps['classifier'].coef_[0])
        else:
            print("⚠️ Model doesn't have feature importance attribute")
            return
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot
        plt.bar(range(len(importances)), importances[indices])
        plt.title(f'Feature Importance - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        
        # Add feature names on x-axis
        feature_names = [feature_columns[i] for i in indices]
        plt.xticks(range(len(importances)), feature_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(importances[indices]):
            plt.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "results/feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Feature importance plot saved to {plot_path}")
        return plot_path
        
    except Exception as e:
        print(f"⚠️ Error creating feature importance plot: {e}")
        return None

def create_model_comparison_plot(results, best_model_name):
    """Create dan save model comparison plot"""
    print("📊 Creating model comparison plot...")
    try:
        # Filter out models with errors
        valid_results = {name: result for name, result in results.items() 
                        if 'error' not in result}
        
        if not valid_results:
            print("⚠️ No valid results for comparison plot")
            return None
        
        models = list(valid_results.keys())
        accuracies = [valid_results[model]['accuracy'] for model in models]
        f1_scores = [valid_results[model]['f1_weighted'] for model in models]
        cv_means = [valid_results[model]['cv_mean'] for model in models]
        cv_stds = [valid_results[model]['cv_std'] for model in models]
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Accuracy comparison
        colors = ['gold' if model == best_model_name else 'lightblue' for model in models]
        bars1 = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (model, bar) in enumerate(zip(models, bars1)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{accuracies[i]:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Test Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Cross-validation scores with error bars
        bars2 = ax2.bar(models, cv_means, yerr=cv_stds, capsize=5, 
                       color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (model, bar) in enumerate(zip(models, bars2)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + cv_stds[i] + 0.005,
                    f'{cv_means[i]:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylabel('Cross-Validation Accuracy', fontsize=12)
        ax2.set_title('Cross-Validation Performance\n(Error bars show ±1 std dev)', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "results/model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Model comparison plot saved to {plot_path}")
        return plot_path
        
    except Exception as e:
        print(f"⚠️ Error creating model comparison plot: {e}")
        return None

def create_model_results_plot(y_test, y_pred, model_name, results):
    """Create dan save model results plot (confusion matrix + metrics)"""
    print("📊 Creating model results plot...")
    try:
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        
        # Plot 2: Performance Metrics Bar Chart
        metrics = ['Accuracy', 'F1 Weighted', 'F1 Macro', 'CV Mean']
        values = [
            results['accuracy'],
            results['f1_weighted'], 
            results['f1_macro'],
            results['cv_mean']
        ]
        
        bars = ax2.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'],
                      edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title(f'Performance Metrics - {model_name}', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "results/model_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Model results plot saved to {plot_path}")
        return plot_path
        
    except Exception as e:
        print(f"⚠️ Error creating model results plot: {e}")
        return None

def save_all_visualizations(best_model, best_model_name, results, feature_columns, y_test, y_pred):
    """Save semua visualizations"""
    print("🎨 Creating and saving all visualizations...")
    
    visualization_paths = {}
    
    # 1. Feature Importance Plot
    feature_plot_path = create_feature_importance_plot(best_model, feature_columns, best_model_name)
    if feature_plot_path:
        visualization_paths['feature_importance'] = feature_plot_path
    
    # 2. Model Comparison Plot
    comparison_plot_path = create_model_comparison_plot(results, best_model_name)
    if comparison_plot_path:
        visualization_paths['model_comparison'] = comparison_plot_path
    
    # 3. Model Results Plot
    results_plot_path = create_model_results_plot(y_test, y_pred, best_model_name, results[best_model_name])
    if results_plot_path:
        visualization_paths['model_results'] = results_plot_path
    
    return visualization_paths

def save_model_and_metadata(best_model, best_model_name, results, feature_columns, encoders):
    """Save model dan metadata dengan proper serialization"""
    print("💾 Saving model and metadata...")
    
    try:
        # Save model menggunakan skops
        model_path = "model/mental_health_pipeline.skops"
        sio.dump(best_model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # ✅ Fix: Convert encoders untuk JSON serialization
        encoders_serializable = {}
        for key, encoder in encoders.items():
            if hasattr(encoder, 'classes_'):
                encoders_serializable[key] = {
                    'classes_': encoder.classes_.tolist(),
                    'type': 'LabelEncoder'
                }
            else:
                encoders_serializable[key] = str(encoder)
        
        # Save encoders menggunakan pickle (bukan JSON)
        encoders_path = "model/encoders.pkl"
        with open(encoders_path, "wb") as f:
            pickle.dump(encoders, f)
        print(f"✅ Encoders saved: {encoders_path}")
        
        # Save feature columns
        feature_path = "model/feature_columns.pkl"
        with open(feature_path, "wb") as f:
            pickle.dump(feature_columns, f)
        print(f"✅ Feature columns saved: {feature_path}")
        
        # ✅ Create JSON-serializable metadata
        metadata = {
            'best_model_name': best_model_name,
            'feature_columns': feature_columns,
            'test_accuracy': float(results[best_model_name]['accuracy']),
            'test_f1': float(results[best_model_name]['f1_weighted']),
            'cv_mean': float(results[best_model_name]['cv_mean']),
            'cv_std': float(results[best_model_name]['cv_std']),
            'timestamp': datetime.now().isoformat(),
            'encoders_info': encoders_serializable  # JSON-serializable version
        }
        
        # Save metadata sebagai JSON
        metadata_path = "model/model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved: {metadata_path}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Error saving model artifacts: {e}")
        import traceback
        traceback.print_exc()
        raise

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
        "metadata_path": "models/model_metadata.json"
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
     # ✅ Import fungsi monitoring dari scripts
    try:
        from scripts.evidently_monitoring import run_evidently_monitoring
        EVIDENTLY_AVAILABLE = True
    except ImportError:
        print("⚠️ Evidently monitoring script not found")
        EVIDENTLY_AVAILABLE = False
    
    print("🚀 Starting ML Training Pipeline with Evidently Integration")
    print("=" * 60)
    
    try:
        # ✅ Setup directories
        os.makedirs("model", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("explanations", exist_ok=True)
        os.makedirs("monitoring/evidently_reports", exist_ok=True)
        print("✅ Directories setup completed")
        
        # ✅ Load and prepare data (menggunakan fungsi yang sudah ada)
        df, encoders = load_and_prepare_data()
        data_source = "mental_health_lite"
        
        # ✅ Skip Evidently monitoring jika ada masalah (dengan proper check)
        if EVIDENTLY_AVAILABLE:
            try:
                numeric_df = df.select_dtypes(include=['number'])
                if len(numeric_df.columns) > 0:
                    run_evidently_monitoring()
                    print("✅ Evidently monitoring completed")
                else:
                    print("⚠️ No numeric columns for Evidently monitoring")
            except Exception as e:
                print(f"⚠️ Evidently monitoring skipped: {e}")
        else:
            print("⚠️ Evidently monitoring not available")
        
        # ✅ Prepare data for training (menggunakan fungsi yang sudah ada)
        X, y, feature_columns = prepare_data_for_training(df, encoders)
        target_column = 'risk_encoded'  # Target sudah ditentukan dalam prepare_data_for_training
        
        print(f"✅ Using target column: {target_column}")
        print(f"📊 Target distribution:\n{y.value_counts()}")
        print(f"📊 Training data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y  # Ubah ke 0.3 sesuai implementasi sebelumnya
        )
        
        print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")
        
        # ✅ Train models (menggunakan fungsi yang sudah ada)
        best_model, best_model_name, results, y_pred = train_models_fixed(
            X_train, X_test, y_train, y_test, feature_columns
        )
        
        # ✅ Save model and metadata (menggunakan fungsi yang sudah ada)
        metadata = save_model_and_metadata(
            best_model, best_model_name, results, feature_columns, encoders
        )
        
        # ✅ Create and save all visualizations (menggunakan fungsi yang sudah ada)
        visualization_paths = save_all_visualizations(
            best_model, best_model_name, results, feature_columns, y_test, y_pred
        )
        
        # ✅ Save metrics to results/metrics.txt
        metrics_txt_path = "results/metrics.txt"
        with open(metrics_txt_path, "w") as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"CV Accuracy: {results[best_model_name]['cv_mean']:.4f}\n")
            f.write(f"Test Accuracy: {results[best_model_name]['accuracy']:.4f}\n")
            f.write(f"Test F1 Score: {results[best_model_name]['f1_weighted']:.4f}\n\n")

            f.write("Model Comparison:\n")
            for model_name, model_result in results.items():
                if 'error' not in model_result:
                    f.write(f"{model_name}: {model_result['cv_mean']:.4f} (+/- {model_result['cv_std']:.4f})\n")

            f.write("\nClassification Report:\n")
            report = classification_report(y_test, y_pred, digits=4)
            f.write(report)
        print(f"✅ Metrics saved to {metrics_txt_path}")


        # Add visualization paths to metadata
        metadata['visualization_paths'] = visualization_paths
        
        # ✅ Update metadata file with visualization info (fix path)
        metadata_path = "model/model_metadata.json"  # Ubah dari "models/" ke "model/"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # ✅ Create training summary (simplified)
        training_summary = {
            "training_completed": True,
            "timestamp": datetime.now().isoformat(),
            "best_model": best_model_name,
            "best_accuracy": metadata.get('test_accuracy', 0.0),
            "data_source": data_source,
            "feature_count": len(feature_columns),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "visualization_paths": visualization_paths,
            "status": "success"
        }
        
        with open("monitoring/training_summary.json", "w") as f:
            json.dump(training_summary, f, indent=2)
        
        with open("results/training_results.json", "w") as f:
            json.dump(training_summary, f, indent=2)
        
        print("=" * 60)
        print("🎉 Training pipeline completed successfully!")
        print(f"🏆 Best model: {best_model_name}")
        print(f"📊 Best accuracy: {metadata.get('test_accuracy', 0.0):.4f}")
        print("🎨 Visualizations created:")
        for viz_type, path in visualization_paths.items():
            print(f"   - {viz_type}: {path}")
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
        # ✅ Create error summary dengan proper directory creation
        os.makedirs("monitoring", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
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


