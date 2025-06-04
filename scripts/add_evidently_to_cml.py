import json
import os

def add_evidently_summary_to_report():
    """Add Evidently summary to CML report dengan better error handling"""
    
    try:
        with open('report.md', 'a') as f:
            f.write("\n## 🔍 Evidently Data Monitoring Results\n\n")
            
            if os.path.exists('monitoring/evidently_summary.json'):
                with open('monitoring/evidently_summary.json', 'r') as summary_file:
                    summary = json.load(summary_file)
                
                # Handle missing or invalid data sizes
                ref_size = summary.get('reference_data_size', 'Unknown')
                curr_size = summary.get('current_data_size', 'Unknown')
                
                # Convert to string dengan proper formatting
                if isinstance(ref_size, (int, float)) and ref_size > 0:
                    ref_size_str = f"{int(ref_size)} rows"
                else:
                    ref_size_str = "Unknown rows"
                
                if isinstance(curr_size, (int, float)) and curr_size > 0:
                    curr_size_str = f"{int(curr_size)} rows"
                else:
                    curr_size_str = "Unknown rows"
                
                f.write(f"**Monitoring Timestamp:** {summary.get('timestamp', 'Unknown')}\n")
                f.write(f"**Reference Data Size:** {ref_size_str}\n")
                f.write(f"**Current Data Size:** {curr_size_str}\n")
                
                drift_status = "🚨 Yes" if summary.get('drift_detected', False) else "✅ No"
                f.write(f"**Data Drift Detected:** {drift_status}\n")
                
                # Add monitoring status if available
                if 'monitoring_status' in summary:
                    status_emoji = "✅" if summary['monitoring_status'] == "success" else "⚠️"
                    f.write(f"**Monitoring Status:** {status_emoji} {summary['monitoring_status'].title()}\n")
                
                if summary.get('error'):
                    f.write(f"**Error:** {summary['error']}\n")
                
                if summary.get('drifted_columns'):
                    f.write("\n**Drifted Columns:**\n")
                    for col_info in summary['drifted_columns']:
                        f.write(f"- {col_info['column']}: drift score {col_info['drift_score']:.3f}\n")
                else:
                    f.write("**Drifted Columns:** None detected\n")
                
                f.write(f"\n**Total Drifted Features:** {len(summary.get('drifted_columns', []))}\n")
                
            else:
                f.write("**Evidently Summary:** Not available - monitoring file not found\n")
            
            f.write("\n")
            
            # Debug information
            f.write("## 🔧 Debug Information\n")
            f.write(f"**Summary File Exists:** {'✅ Yes' if os.path.exists('monitoring/evidently_summary.json') else '❌ No'}\n")
            if os.path.exists('monitoring/evidently_summary.json'):
                try:
                    with open('monitoring/evidently_summary.json', 'r') as debug_file:
                        debug_content = debug_file.read()
                    f.write(f"**Summary File Size:** {len(debug_content)} characters\n")
                except:
                    f.write("**Summary File Size:** Could not read\n")
            f.write("\n")
        
        print("✅ Evidently summary added to CML report with debug info")
        
    except Exception as e:
        print(f"⚠️ Error adding Evidently summary to CML report: {e}")
        try:
            with open('report.md', 'a') as f:
                f.write("\n## 🔍 Evidently Data Monitoring Results\n\n")
                f.write("**Status:** Monitoring failed to generate summary\n")
                f.write(f"**Error:** {str(e)}\n\n")
        except:
            pass

if __name__ == "__main__":
    add_evidently_summary_to_report()
