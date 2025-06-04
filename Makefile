install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

monitor:
	python monitor.py

eval:
	echo "## Multi-Model Training Results" > report.md
	echo "" >> report.md
	echo "### Model Comparison" >> report.md
	if [ -f "./results/model_comparison.json" ]; then \
		python -c "import json; data=json.load(open('./results/model_comparison.json')); print(f'**Best Model:** {data[\"best_model\"]}'); [print(f'- {k}: {v[\"mean_accuracy\"]:.4f}') for k,v in data['model_scores'].items()]" >> report.md; \
	fi
	echo "" >> report.md
	echo "### Detailed Metrics" >> report.md
	cat ./results/metrics.txt >> report.md
	echo "" >> report.md
	echo "### Model Performance Visualization" >> report.md
	echo "![Model Comparison](./results/model_comparison.png)" >> report.md
	echo "" >> report.md
	echo "### Confusion Matrix" >> report.md
	echo "![Confusion Matrix](./results/model_results.png)" >> report.md
	echo "" >> report.md
	echo "### SHAP Explanations" >> report.md
	echo "![SHAP Summary](./results/shap_summary.png)" >> report.md
	echo "![SHAP Importance](./results/shap_importance.png)" >> report.md
	cml comment create report.md

check-drift:
	python -c "
	import pandas as pd
	from monitor import detect_data_drift
	df = pd.read_csv('data/mental_health_lite.csv')
	drift, report = detect_data_drift('monitoring/whylogs_profiles/training_data_profile', df)
	if drift:
		print('Data drift detected! Triggering retraining...')
		exit(1)
	else:
		print('No significant data drift detected.')
		exit(0)
	"

retrain-if-drift: check-drift train

hf-login:
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload $(HF_USERNAME)/Mental-Health-Risk-Prediction ./app --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload $(HF_USERNAME)/Mental-Health-Risk-Prediction ./model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload $(HF_USERNAME)/Mental-Health-Risk-Prediction ./results --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub
