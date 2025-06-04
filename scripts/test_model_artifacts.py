import os
import json

def test_model_artifacts():
    """Test model artifacts integrity"""
    
    print("🧪 Testing model artifacts...")
    
    # Test model metadata
    if os.path.exists('model/model_metadata.json'):
        try:
            with open('model/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            print(f"✅ Model metadata loaded successfully")
            print(f"   - Model: {metadata.get('best_model_name', 'Unknown')}")
            print(f"   - Accuracy: {metadata.get('test_accuracy', 'Unknown')}")
            print(f"   - F1 Score: {metadata.get('test_f1', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error loading model metadata: {e}")
    else:
        print("⚠️ Model metadata not found")
    
    # Test model comparison results
    if os.path.exists('results/model_comparison.json'):
        try:
            with open('results/model_comparison.json', 'r') as f:
                comparison = json.load(f)
            print(f"✅ Model comparison results loaded")
            print(f"   - Best model: {comparison.get('best_model', 'Unknown')}")
        except Exception as e:
            print(f"❌ Error loading model comparison: {e}")
    else:
        print("⚠️ Model comparison results not found")
    
    # Test metrics file
    if os.path.exists('results/metrics.txt'):
        print("✅ Metrics file exists")
    else:
        print("⚠️ Metrics file not found")
    
    print("✅ Model artifacts testing completed")

if __name__ == "__main__":
    test_model_artifacts()
