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
    best_model_name = max(
        [name for name in results.keys() if 'error' not in results[name]], 
        key=lambda x: results[x]['accuracy']
    )
    
    best_model = trained_models[best_model_name]
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"🏆 Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    return best_model, best_model_name, results

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
        best_model, best_model_name, results = train_models(
            X_train, X_test, y_train, y_test, feature_columns
        )
        
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
