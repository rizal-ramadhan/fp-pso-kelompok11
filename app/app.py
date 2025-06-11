import gradio as gr
import pandas as pd
import numpy as np
import pickle
import json
import os
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import skops.io as sio
from datetime import datetime
from skops.io import get_untrusted_types, load
import tempfile
import warnings
warnings.filterwarnings('ignore')

class MentalHealthPredictor:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_columns = None
        self.model_metadata = None
        self.explainer = None
        self.load_model_artifacts()

    def load_model_artifacts(self):
        try:
            model_path = "model/mental_health_pipeline.skops"
            if os.path.exists(model_path):
                untrusted_types = get_untrusted_types(file=model_path)
                self.model = load(model_path, trusted=untrusted_types)
                print("✅ Model loaded successfully")
            else:
                print(f"❌ Model file not found at {model_path}")
                return False

            encoders_path = "model/encoders.pkl"
            if os.path.exists(encoders_path):
                with open(encoders_path, "rb") as f:
                    self.encoders = pickle.load(f)
                print("✅ Encoders loaded successfully")

            feature_path = "model/feature_columns.pkl"
            if os.path.exists(feature_path):
                with open(feature_path, "rb") as f:
                    self.feature_columns = pickle.load(f)
                print("✅ Feature columns loaded successfully")

            metadata_path = "model/model_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.model_metadata = json.load(f)
                print("✅ Model metadata loaded successfully")

            return True

        except Exception as e:
            print(f"❌ Error loading model artifacts: {e}")
            return False

    def prepare_input_data(self, age, gender, employment, work_env, mental_history, 
                          seeks_treatment, stress_level, sleep_hours, physical_activity, 
                          depression_score, anxiety_score, social_support, productivity):
        try:
            input_data = {
                'age': float(age),
                'stress_level': float(stress_level),
                'sleep_hours': float(sleep_hours),
                'physical_activity_days': float(physical_activity),
                'depression_score': float(depression_score),
                'anxiety_score': float(anxiety_score),
                'social_support_score': float(social_support),
                'productivity_score': float(productivity),
                'gender': gender,
                'employment_status': employment,
                'work_environment': work_env,
                'mental_health_history': mental_history,
                'seeks_treatment': seeks_treatment
            }

            df = pd.DataFrame([input_data])

            if self.encoders:
                categorical_cols = ['gender', 'employment_status', 'work_environment', 
                                    'mental_health_history', 'seeks_treatment']

                for col in categorical_cols:
                    if col in self.encoders and col in df.columns:
                        encoder = self.encoders[col]
                        value = df[col].iloc[0]
                        if value in encoder.classes_:
                            df[f'{col}_encoded'] = encoder.transform(df[col])
                        else:
                            df[f'{col}_encoded'] = 0
                            print(f"⚠️ Unknown value '{value}' for {col}, using default")

            if self.feature_columns:
                available_features = [col for col in self.feature_columns if col in df.columns]
                df_features = df[available_features]
            else:
                df_features = df.select_dtypes(include=[np.number])

            return df_features

        except Exception as e:
            print(f"❌ Error preparing input data: {e}")
            return None

    def get_risk_interpretation(self, risk_level, probabilities):
        interpretations = {
            "Low": {
                "emoji": "🟢",
                "color": "#28a745",
                "description": "Your responses suggest a lower likelihood of mental health concerns.",
                "recommendations": [
                    "Maintain healthy lifestyle habits",
                    "Keep regular physical activity and sleep routine",
                    "Stay socially connected",
                    "Practice preventive stress management"
                ]
            },
            "Medium": {
                "emoji": "🟡",
                "color": "#ffc107",
                "description": "Some areas may benefit from attention.",
                "recommendations": [
                    "Consider speaking with a professional",
                    "Reduce stress via meditation or hobbies",
                    "Focus on sleep and physical activity",
                    "Monitor emotional well-being regularly"
                ]
            },
            "High": {
                "emoji": "🔴",
                "color": "#dc3545",
                "description": "You may benefit from professional mental health support.",
                "recommendations": [
                    "🚨 Contact a mental health professional soon",
                    "Seek support from friends or crisis services",
                    "Practice immediate stress relief techniques",
                    "Consider medical evaluation"
                ]
            }
        }
        return interpretations.get(risk_level, interpretations["Medium"])

    def generate_shap_explanation(self, input_features):
        try:
            feature_names = input_features.columns.tolist()
            feature_values = input_features.iloc[0].values

            plt.figure(figsize=(10, 8))
            y_pos = np.arange(len(feature_names))
            colors = ['#ff7f7f' if val > np.median(feature_values) else '#7fbfff' for val in feature_values]

            bars = plt.barh(y_pos, feature_values, color=colors, alpha=0.7, edgecolor='black')
            plt.yticks(y_pos, [name.replace('_encoded', '').replace('_', ' ').title() for name in feature_names])
            plt.xlabel('Feature Values')
            plt.title('Feature Contribution Overview', fontsize=14)
            plt.grid(axis='x', alpha=0.3)

            for i, (bar, val) in enumerate(zip(bars, feature_values)):
                plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center')

            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                plt.savefig(tmpfile.name, format='png', dpi=150, bbox_inches='tight')
                return tmpfile.name

        except Exception as e:
            print(f"⚠️ SHAP explanation error: {e}")
            return None

    def predict_mental_health_risk(self, age, gender, employment, work_env, mental_history, 
                                   seeks_treatment, stress_level, sleep_hours, physical_activity, 
                                   depression_score, anxiety_score, social_support, productivity):
        if not self.model:
            return self.create_fallback_response()

        try:
            input_features = self.prepare_input_data(
                age, gender, employment, work_env, mental_history, 
                seeks_treatment, stress_level, sleep_hours, physical_activity, 
                depression_score, anxiety_score, social_support, productivity
            )

            if input_features is None:
                return self.create_fallback_response()

            prediction_proba = self.model.predict_proba(input_features)[0]
            prediction = self.model.predict(input_features)[0]

            risk_levels = ['Low', 'Medium', 'High']
            risk_prediction = risk_levels[prediction] if prediction < len(risk_levels) else 'Medium'

            risk_info = self.get_risk_interpretation(risk_prediction, prediction_proba)

            # ======= RESULT HTML START =======
            result_html = f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin: 20px 0;">
                <h2 style="margin: 0; font-size: 2em;">{risk_info['emoji']} Mental Health Risk Assessment</h2>
                <h3 style="margin: 10px 0; color: {risk_info['color']}; background: white; padding: 10px; border-radius: 5px;">
                    Risk Level: {risk_prediction}
                </h3>
            </div>

            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; color: #222;">
                <h3 style="color: #2d3a4a; font-weight: bold;">📊 Probability Scores:</h3>
                <div style="margin: 15px 0;">
            """

            for i, (level, prob) in enumerate(zip(risk_levels, prediction_proba)):
                emoji = "🟢" if level == "Low" else "🟡" if level == "Medium" else "🔴"
                width = prob * 100
                bar_color = "#28a745" if level == "Low" else "#ffc107" if level == "Medium" else "#dc3545"
                # Perbaikan: Gunakan warna teks yang lebih kontras
                text_color = "white" if level in ["Low", "High"] else "#212529"  # Putih untuk hijau/merah, hitam untuk kuning
                font_size = "0.95em" if width > 10 else "0.85em"
            
                result_html += f"""
                    <div style="margin: 10px 0;">
                        <strong style="color: #212529;">{emoji} {level} Risk:</strong> {prob:.1%}
                        <div style="background: #e9ecef; border-radius: 10px; height: 25px; margin: 5px 0;">
                            <div style="background: {bar_color}; 
                                        height: 100%; width: {width}%; border-radius: 10px; 
                                        display: flex; align-items: center; justify-content: center; 
                                        color: {text_color}; font-weight: bold; font-size: {font_size};">
                                {prob:.1%}
                            </div>
                        </div>
                    </div>
                """

            result_html += f"""
                </div>
            </div>

            <div style="background: #fff8e1; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 6px solid {risk_info['color']};">
                <h3 style="color: #333;">🧠 Interpretation:</h3>
                <p style="font-size: 1.1em; line-height: 1.6; color: #333;">{risk_info['description']}</p>
            </div>
            """

            if self.model_metadata:
                result_html += f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #007bff;">
                    <h4 style="color: #212529; font-weight: bold;">🤖 Model Information:</h4>
                    <ul style="margin: 10px 0; color: #212529;">
                        <li><strong style="color:#000000;">Model:</strong> {self.model_metadata.get('best_model_name', 'Unknown')}</li>
                        <li><strong style="color:#000000;">Accuracy:</strong> {self.model_metadata.get('test_accuracy', 0):.1%}</li>
                        <li><strong style="color:#000000;">Last Updated:</strong> {self.model_metadata.get('timestamp', 'Unknown')[:10]}</li>
                    </ul>
                </div>
                """

            # ======= RESULT HTML END =======

            explanation_plot = self.generate_shap_explanation(input_features)
            recommendations = self.generate_recommendations(risk_info)

            return result_html, explanation_plot, recommendations

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self.create_fallback_response()

    def create_fallback_response(self):
        result = """
        <div style="text-align: center; padding: 20px; border-radius: 10px; background: #ffc107; color: #212529;">
            <h2>⚠️ Assessment Temporarily Unavailable</h2>
            <p>We're experiencing technical difficulties. Please try again later or consult a healthcare professional.</p>
        </div>
        """
        return result, None, self.get_general_recommendations()

    def get_general_recommendations(self):
        return """
        <div style="background: white; padding: 25px; border-radius: 10px;">
            <h2>💡 General Mental Health Tips</h2>
            <ul style="line-height: 1.8;">
                <li>Maintain regular sleep schedule (7-9 hours per night)</li>
                <li>Exercise regularly (at least 30 minutes, 3-4 times per week)</li>
                <li>Practice stress management techniques</li>
                <li>Stay connected with friends and family</li>
                <li>Seek professional help when needed</li>
            </ul>
        </div>
        """

    def generate_recommendations(self, risk_info):
        recommendations_html = f"""
        <div style="background: #f8f9fa; color: #212529; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <h2 style="color: {risk_info['color']}; margin-bottom: 20px;">💡 Personalized Recommendations</h2>
            <div style="background: #ffffff; color: #212529; padding: 20px; border-radius: 8px; margin: 15px 0; border: 1px solid #dee2e6;">
                <h3 style="color: {risk_info['color']}; margin-top: 0;">Based on your {risk_info['emoji']} risk assessment:</h3>
                <ul style="line-height: 1.8; font-size: 1.1em; color: #212529;">
        """
        for rec in risk_info['recommendations']:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += """
                </ul>
            </div>
            <div style="background: #fff3cd; color: #856404; padding: 15px; border-radius: 8px; border: 1px solid #ffeaa7;">
                <p style="margin: 0; font-style: italic; text-align: center; color: #856404;">
                    <strong>⚠️ Important:</strong> This assessment is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment.
                </p>
            </div>
        </div>
        """
        return recommendations_html

# Inisialisasi
predictor = MentalHealthPredictor()

def predict_interface(age, gender, employment, work_env, mental_history, seeks_treatment,
                      stress_level, sleep_hours, physical_activity, depression_score, 
                      anxiety_score, social_support, productivity):
    return predictor.predict_mental_health_risk(
        age, gender, employment, work_env, mental_history, seeks_treatment,
        stress_level, sleep_hours, physical_activity, depression_score, 
        anxiety_score, social_support, productivity
    )

with gr.Blocks(
    title="Mental Health Risk Identifier", 
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    /* Perbaikan kontras untuk elemen teks */
    .gradio-container .prose {
        color: #212529 !important;
    }
    .gradio-container p {
        color: #212529 !important;
    }
    .gradio-container li {
        color: #212529 !important;
    }
    
    /* Styling dropdown yang lebih spesifik dan kuat */
    .gradio-container .gr-form > label > select,
    .gradio-container select,
    .gradio-container .dropdown select,
    select {
        color: #ffffff !important;
        background-color: #374151 !important;
        border: 1px solid #6b7280 !important;
    }
    
    .gradio-container .gr-form > label > select option,
    .gradio-container select option,
    select option {
        color: #ffffff !important;
        background-color: #374151 !important;
    }
    
    /* Target lebih spesifik untuk Gradio dropdown */
    .gradio-container .gr-box select,
    .gradio-container .gr-padded select {
        color: #ffffff !important;
        background-color: #374151 !important;
    }
    """
) as app:

    gr.HTML("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5em;">🧠 Mental Health Risk Identifier</h1>
        <p style="margin: 1rem 0 0 0; font-size: 1.2em; opacity: 0.9;">
            AI-powered mental health risk assessment with explainable predictions
        </p>
    </div>
    """)

    gr.Markdown("### 📝 Fill out the form below to get your personalized risk assessment.")

    with gr.Row():
        with gr.Column(scale=1):
            age = gr.Number(label="Age", value=30, minimum=18, maximum=80, precision=0)
            gender = gr.Dropdown(["Male", "Female", "Non-binary", "Prefer not to say"], label="Gender")
            employment = gr.Dropdown(["Employed", "Unemployed", "Student", "Self-employed"], label="Employment")
            work_env = gr.Dropdown(["On-site", "Remote", "Hybrid"], label="Work Environment")
            mental_history = gr.Dropdown(["Yes", "No"], label="Mental Health History")
            seeks_treatment = gr.Dropdown(["Yes", "No"], label="Currently Seeking Treatment")

        with gr.Column(scale=1):
            stress_level = gr.Slider(1, 10, value=5, label="Stress Level")
            sleep_hours = gr.Slider(3, 12, value=7, step=0.5, label="Sleep Hours")
            physical_activity = gr.Slider(0, 7, value=3, step=1, label="Physical Activity Days")
            depression_score = gr.Slider(0, 50, value=10, label="Depression Score")
            anxiety_score = gr.Slider(0, 50, value=10, label="Anxiety Score")
            social_support = gr.Slider(0, 100, value=70, step=5, label="Social Support Score")
            productivity = gr.Slider(0, 100, value=70, step=5, label="Productivity Score")

    with gr.Row():
        predict_btn = gr.Button("🎯 Assess Mental Health Risk", variant="primary")
        clear_btn = gr.Button("🔄 Clear Form", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            prediction_output = gr.HTML()
            recommendations_output = gr.HTML()
        with gr.Column(scale=1):
            explanation_output = gr.Image(type="filepath", height=400)

    predict_btn.click(
        fn=predict_interface,
        inputs=[age, gender, employment, work_env, mental_history, seeks_treatment,
                stress_level, sleep_hours, physical_activity, depression_score, 
                anxiety_score, social_support, productivity],
        outputs=[prediction_output, explanation_output, recommendations_output]
    )

    clear_btn.click(
        fn=lambda: [30, "Male", "Employed", "On-site", "No", "No", 5, 7, 3, 10, 10, 70, 70],
        outputs=[age, gender, employment, work_env, mental_history, seeks_treatment,
                 stress_level, sleep_hours, physical_activity, depression_score, 
                 anxiety_score, social_support, productivity]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)

