import os
import json

def test_evidently_integration():
    """Test Evidently integration"""
    
    print("🔍 Testing Evidently integration...")
    
    # Test Evidently summary
    if os.path.exists('monitoring/evidently_summary.json'):
        try:
            with open('monitoring/evidently_summary.json', 'r') as f:
                summary = json.load(f)
            
            print("✅ Evidently summary loaded successfully")
            print(f"   - Drift detected: {summary.get('drift_detected', 'Unknown')}")
            print(f"   - Drifted columns: {len(summary.get('drifted_columns', []))}")
            
        except Exception as e:
            print(f"❌ Error loading Evidently summary: {e}")
    else:
        print("⚠️ Evidently summary not found")
    
    # Test Evidently reports directory
    if os.path.exists('monitoring/evidently_reports'):
        reports = os.listdir('monitoring/evidently_reports')
        html_reports = [r for r in reports if r.endswith('.html')]
        
        if html_reports:
            print(f"✅ Found {len(html_reports)} Evidently HTML reports")
            for report in html_reports:
                print(f"   - {report}")
        else:
            print("⚠️ No HTML reports found in evidently_reports directory")
    else:
        print("⚠️ Evidently reports directory not found")
    
    # Test reference data
    if os.path.exists('monitoring/reference_data.csv'):
        print("✅ Reference data file exists")
    else:
        print("⚠️ Reference data file not found")
    
    print("✅ Evidently integration testing completed")

if __name__ == "__main__":
    test_evidently_integration()
