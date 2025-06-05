# 🧠 Mental Health Risk Prediction MLOps

[![CI Pipeline](https://github.com/rizal-ramadhan/fp-pso-kelompok11/actions/workflows/ci.yml/badge.svg)](https://github.com/rizal-ramadhan/fp-pso-kelompok11/actions/workflows/ci.yml)
[![CD Pipeline](https://github.com/rizal-ramadhan/fp-pso-kelompok11/actions/workflows/cd.yml/badge.svg)](https://github.com/rizal-ramadhan/fp-pso-kelompok11/actions/workflows/cd.yml)
[![MLOps Level](https://img.shields.io/badge/MLOps-Level%202-green)](https://ml-ops.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Advanced MLOps implementation for mental health risk assessment using multi-model ensemble and comprehensive monitoring**

## 🎯 Overview

This project implements a production-ready MLOps pipeline for mental health risk prediction, featuring automated training, deployment, and monitoring. The system uses multiple machine learning algorithms with explainable AI capabilities, designed specifically for healthcare applications requiring high reliability and interpretability.

### Key Features

- 🤖 **Multi-Model Ensemble**: RandomForest, XGBoost, and LightGBM with automatic selection
- 🔍 **Evidently AI Monitoring**: Real-time data quality and drift detection
- 📊 **SHAP Explanations**: Model interpretability for healthcare compliance
- 🚀 **CI/CD Pipeline**: Automated testing, training, and deployment
- 🌐 **Production Deployment**: Gradio app hosted on Hugging Face Spaces
- 📈 **CML Integration**: Automated reporting with visual analytics

## 🏗️ Architecture

### MLOps Pipeline Overview
![image](https://github.com/user-attachments/assets/532a7655-d6f6-4f26-8b27-2684bf512cf6)

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **ML Framework** | Scikit-learn, XGBoost, LightGBM | Model training and inference |
| **Monitoring** | Evidently AI, SHAP | Data quality and model explainability |
| **CI/CD** | GitHub Actions, CML | Automated testing and deployment |
| **Deployment** | Hugging Face Spaces, Gradio | Model serving and user interface |
| **Data Processing** | Pandas, NumPy | Data manipulation and analysis |
| **Visualization** | Matplotlib, Seaborn | Performance charts and reports |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Git
- GitHub account
- Hugging Face account (for deployment)

### Local Setup

1. **Clone the repository**
git clone https://github.com/rizal-ramadhan/fp-pso-kelompok11.git
cd fp-pso-kelompok11

2. **Install dependencies**
pip install -r requirements.txt

3. **Run training pipeline**
python train.py or use notebook.ipynb for further analysis

4. **Start local app**
cd app
python App.py


### Production Deployment

1. **Set up GitHub Secrets**
HF_TOKEN: your_huggingface_token
HF_USERNAME: your_huggingface_username

2. **Push to main branch**
git push origin main


3. **Monitor deployment**
- CI Pipeline: Automated testing and training
- CD Pipeline: Deployment to Hugging Face Spaces

## 🤖 Machine Learning Pipeline

### Data Processing

- **Feature Engineering**: Automated encoding of categorical variables
- **Data Validation**: Schema validation and quality checks
- **Drift Detection**: Evidently AI monitoring for distribution changes
- **Quality Gates**: Automated data quality thresholds

### Model Training

Multi-model training with automatic selection
models = {
'RandomForest': RandomForestClassifier(...),
'XGBoost': XGBClassifier(...),
'LightGBM': LGBMClassifier(...)
}

Cross-validation and selection
best_model = select_best_model(models, X_train, y_train)


### Model Evaluation

- **Cross-Validation**: 5-fold stratified validation
- **Performance Metrics**: Accuracy, F1-score, Precision, Recall
- **Quality Thresholds**: Minimum 70% accuracy requirement
- **SHAP Analysis**: Feature importance and model interpretability

## 📊 Monitoring & Observability

### Data Monitoring

- **Evidently AI Integration**: Real-time data quality monitoring
- **Drift Detection**: Statistical tests for distribution changes
- **Data Quality Reports**: Automated HTML reports generation
- **Alert System**: Automated retraining triggers

### Model Monitoring

- **Performance Tracking**: Accuracy and F1-score monitoring
- **SHAP Explanations**: Real-time model interpretability
- **Prediction Monitoring**: Input/output validation
- **Usage Analytics**: Application performance metrics

### Example Monitoring Output

{
"timestamp": "2024-01-15T10:30:00Z",
"reference_data_size": 700,
"current_data_size": 300,
"drift_detected": false,
"drifted_columns": [],
"monitoring_status": "success"
}


## 🔄 CI/CD Pipeline

### Continuous Integration (CI)

1. **Code Quality** 
   - Black formatting check
   - Flake8 linting
   - Code complexity analysis

2. **Data Validation** 
   - Schema validation
   - Quality checks
   - Evidently monitoring

3. **Model Training** 
   - Multi-model training
   - Cross-validation
   - Artifact generation

4. **Security Scan** 
   - Dependency scanning
   - Code security analysis

5. **Integration Test** 
   - Artifact validation
   - Functionality testing

6. **Notify Status** 
   - CML report generation
   - Pipeline summary

### Continuous Deployment (CD)

1. **Artifact Validation**: Model and metadata verification
2. **Staging Deployment**: Automated deployment to staging environment
3. **Health Checks**: Application functionality validation
4. **Manual Approval**: Production deployment gate
5. **Production Deployment**: Live application update
6. **Smoke Tests**: Production environment validation

## 🌐 Production Application

### Gradio Interface

The production application provides:

- **Risk Assessment Form**: Comprehensive mental health questionnaire
- **Real-time Predictions**: Instant risk level classification
- **SHAP Explanations**: Visual feature importance analysis
- **Personalized Recommendations**: Tailored mental health guidance
- **Crisis Resources**: Emergency contact information

### Deployment Environments

- **Staging**: https://huggingface.co/spaces/rizalramadhan/mental-health-mlops-staging
- **Production**: https://huggingface.co/spaces/rizalramadhan/mental-health-risk-prediction

### MLOps Maturity

**Level 2: CI/CD Pipeline Automation**

✅ **Achieved:**
- Automated ML pipeline
- Continuous integration
- Continuous deployment
- Model monitoring
- Version control

🔄 **Future Enhancements:**
- A/B testing capabilities
- Advanced monitoring dashboards
- Multi-environment orchestration
- Feature store implementation

## 🔒 Security & Privacy

### Data Protection

- **No Data Storage**: All processing done locally
- **Privacy-First Design**: No user tracking or data collection
- **HIPAA Considerations**: Healthcare privacy guidelines compliance
- **Secure Processing**: Client-side computations only

### Security Measures

- **Dependency Scanning**: Automated vulnerability detection
- **Code Security**: Static analysis with Bandit
- **Secret Management**: GitHub Secrets for sensitive data
- **Access Control**: Role-based permissions

## ⚠️ Important Disclaimers

### Medical Disclaimer

This tool is for **educational and informational purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

### Limitations

- Not a diagnostic tool
- Based on self-reported data
- Cannot replace clinical assessment
- Should not be used for crisis situations

### Crisis Resources

**Emergency Contacts:**
- **National Suicide Prevention Lifeline**: 988
- **Crisis Text Line**: Text HOME to 741741
- **SAMHSA National Helpline**: 1-800-662-4357
- **Emergency Services**: 911

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Ensure CI pipeline passes

## 📄 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mental Health Professionals**: For domain expertise and validation
- **Open Source Community**: For tools and frameworks
- **Healthcare Organizations**: For privacy and compliance guidelines
- **Research Community**: For validation methodologies

## 🔗 Related Links

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [MLOps Principles](https://ml-ops.org/)
- [Mental Health Resources](https://www.nimh.nih.gov/)

---

**Built with ❤️ for mental health awareness and support**

*Remember: Taking care of your mental health is just as important as taking care of your physical health.*

---

### 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/rizal-ramadhan/fp-pso-kelompok11?style=social)
![GitHub forks](https://img.shields.io/github/forks/rizal-ramadhan/fp-pso-kelompok11?style=social)
![GitHub issues](https://img.shields.io/github/issues/rizal-ramadhan/fp-pso-kelompok11)
![GitHub last commit](https://img.shields.io/github/last-commit/rizal-ramadhan/fp-pso-kelompok11)

**Last Updated**: June 2025
