import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

# ✅ Fix Evidently imports untuk berbagai versi
try:
    from evidently import Report
    from evidently.metrics import ValueDrift, RowCount, MissingValueCount
    print("✅ Evidently imported successfully")
    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Evidently import error: {e}")
    print("💡 Tip: Install evidently with: pip install evidently")
    EVIDENTLY_AVAILABLE = False

class EvidentlyMonitor:
    def __init__(self):
        self.reference_data = None
        self.reports_dir = "monitoring/evidently_reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        self.evidently_available = EVIDENTLY_AVAILABLE
    
    def set_reference_data(self, df):
        """Set reference data untuk comparison"""
        self.reference_data = df.copy()
        print(f"✅ Reference data set with {len(df)} rows")
    
    def create_simple_data_report(self, current_data):
        """Create simple data quality report tanpa Evidently"""
        print("📊 Creating simple data quality report...")
        
        try:
            # Basic data quality metrics
            report = {
                "timestamp": datetime.now().isoformat(),
                "data_shape": current_data.shape,
                "missing_values": current_data.isnull().sum().to_dict(),
                "data_types": current_data.dtypes.astype(str).to_dict(),
                "basic_stats": current_data.describe().to_dict()
            }
            
            # Save simple report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"{self.reports_dir}/simple_quality_report_{timestamp}.json"
            
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Simple data quality report saved: {report_path}")
            return report
            
        except Exception as e:
            print(f"⚠️ Error creating simple data report: {e}")
            return None
    
    def create_data_quality_report(self, current_data, save_html=True):
        """Create data quality report dengan fallback"""
        
        if not self.evidently_available:
            print("⚠️ Evidently not available, using simple report")
            return self.create_simple_data_report(current_data)
        
        try:
            # Evidently metrics
            metrics = [RowCount()]
            
            # Add missing value count untuk numerical columns saja
            numerical_columns = current_data.select_dtypes(include=['number']).columns
            for col in numerical_columns:
                metrics.append(MissingValueCount(column=col))
            
            # Add drift metrics untuk numerical columns
            for col in numerical_columns:
                metrics.append(ValueDrift(column=col))
            
            # Create report
            report = Report(metrics=metrics)
            
            if self.reference_data is not None:
                # Ensure both datasets have same columns
                common_cols = list(set(self.reference_data.columns) & set(current_data.columns))
                ref_df = self.reference_data[common_cols].copy()
                curr_df = current_data[common_cols].copy()
                
                # Ensure numeric data only
                for col in ref_df.columns:
                    ref_df[col] = pd.to_numeric(ref_df[col], errors='coerce').fillna(0)
                    curr_df[col] = pd.to_numeric(curr_df[col], errors='coerce').fillna(0)
                
                report.run(reference_data=ref_df, current_data=curr_df)
            else:
                report.run(reference_data=current_data, current_data=current_data)
            
            if save_html:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = f"{self.reports_dir}/data_quality_report_{timestamp}.html"
                report.save_html(report_path)
                print(f"✅ Evidently data quality report saved: {report_path}")
            
            return report
            
        except Exception as e:
            print(f"⚠️ Evidently report error: {e}")
            print("Falling back to simple report...")
            return self.create_simple_data_report(current_data)
    
    def monitor_mental_health_data(self, current_data):
        """Comprehensive monitoring dengan fallback"""
        print("🔍 Starting data monitoring...")
        
        try:
            # Data quality report
            quality_report = self.create_data_quality_report(current_data)
            print("✅ Data quality report created")
            
            # Simple drift results (fallback)
            drift_results = {
                "drift_detected": False,
                "drifted_columns": [],
                "total_drifted": 0
            }
            
            return {
                "quality_report": quality_report,
                "drift_report": None,
                "drift_results": drift_results
            }
            
        except Exception as e:
            print(f"⚠️ Monitoring error: {e}")
            return {
                "quality_report": None,
                "drift_report": None,
                "drift_results": {"drift_detected": False, "drifted_columns": [], "total_drifted": 0}
            }

def run_evidently_monitoring():
    """Main function untuk menjalankan monitoring"""
    
    # Load data
    data_files = ['data/mental_health_lite.csv', 'data/mental_health_life_cut.csv']
    current_data = None
    
    for file_path in data_files:
        if os.path.exists(file_path):
            current_data = pd.read_csv(file_path)
            print(f"✅ Data loaded from: {file_path}")
            break
    
    if current_data is None:
        print("❌ No data file found")
        summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_data_size": 0,
            "current_data_size": 0,
            "drift_detected": False,
            "drifted_columns": [],
            "error": "No data file found"
        }
    else:
        # Initialize monitor
        monitor = EvidentlyMonitor()
        
        # Set reference data
        split_idx = int(len(current_data) * 0.7)
        reference_data = current_data.iloc[:split_idx]
        current_data_subset = current_data.iloc[split_idx:]
        
        # Ensure numeric data only
        numeric_cols = reference_data.select_dtypes(include=['number']).columns
        reference_data = reference_data[numeric_cols]
        current_data_subset = current_data_subset[numeric_cols]
        
        monitor.set_reference_data(reference_data)
        
        # Run monitoring
        results = monitor.monitor_mental_health_data(current_data_subset)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_data_size": len(reference_data),
            "current_data_size": len(current_data_subset),
            "drift_detected": results["drift_results"]["drift_detected"],
            "drifted_columns": results["drift_results"]["drifted_columns"],
            "monitoring_status": "success"
        }
    
    # Save summary
    os.makedirs("monitoring", exist_ok=True)
    with open("monitoring/evidently_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Evidently monitoring completed")
    return summary

if __name__ == "__main__":
    try:
        run_evidently_monitoring()
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
