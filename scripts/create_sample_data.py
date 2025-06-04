import pandas as pd
import numpy as np
import os

def create_sample_dataset():
    os.makedirs('data', exist_ok=True)
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_samples),
        'employment_status': np.random.choice(['Employed', 'Unemployed', 'Student', 'Self-employed'], n_samples),
        'work_environment': np.random.choice(['On-site', 'Remote', 'Hybrid'], n_samples),
        'mental_health_history': np.random.choice(['Yes', 'No'], n_samples),
        'seeks_treatment': np.random.choice(['Yes', 'No'], n_samples),
        'stress_level': np.random.randint(1, 11, n_samples),
        'sleep_hours': np.random.uniform(3, 12, n_samples),
        'physical_activity_days': np.random.randint(0, 8, n_samples),
        'depression_score': np.random.randint(0, 51, n_samples),
        'anxiety_score': np.random.randint(0, 51, n_samples),
        'social_support_score': np.random.randint(0, 101, n_samples),
        'productivity_score': np.random.randint(0, 101, n_samples),
        'mental_health_risk': np.random.choice(['Low', 'Medium', 'High'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/mental_health_lite.csv', index=False)
    print('✅ Sample dataset created')

if __name__ == "__main__":
    create_sample_dataset()
