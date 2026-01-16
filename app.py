"""
================================================================================
EMPLOYABILITY PREDICTOR - STREAMLIT WEB APPLICATION
================================================================================
Anavid Assignment - Student Employability Prediction System
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Employability Predictor - Anavid Assignment",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = Path("style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stButton > button {
            border-radius: 4px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            border: 2px solid #0066cc !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(0, 102, 204, 0.2) !important;
        }
        .stButton > button[kind="primary"] {
            background-color: #0066cc !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

load_css()

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('model.pkl')
        imputer = joblib.load('imputer.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, imputer, feature_names
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

model, imputer, feature_names = load_model_artifacts()

# Feature descriptions
FEATURE_INFO = {
    'Gender': 'Gender of the student (0: Female, 1: Male)',
    'Nationality': 'Nationality (0: Omani, 1: Non-Omani)',
    'Major': 'Major field of study (0: Business, 1: Engineering, 2: IT, etc.)',
    'Level': 'Academic level (0: First Year, 1: Second Year, etc.)',
    'IE1': 'Internship Experience - Industry exposure',
    'IE2': 'Internship Experience - Professional network',
    'IE3': 'Internship Experience - Practical skills',
    'IE4': 'Internship Experience - Job market knowledge',
    'IE5': 'Internship Experience - Career readiness',
    'SMSK1': 'Soft & Management Skills - Communication',
    'SMSK2': 'Soft & Management Skills - Teamwork',
    'SMSK3': 'Soft & Management Skills - Leadership',
    'SMSK4': 'Soft & Management Skills - Problem solving',
    'RAS1': 'Research & Analytical Skills - Critical thinking',
    'RAS2': 'Research & Analytical Skills - Data analysis',
    'RAS3': 'Research & Analytical Skills - Research methodology',
    'RAS4': 'Research & Analytical Skills - Information synthesis',
    'RAS5': 'Research & Analytical Skills - Technical writing',
    'TL1': 'Technology Literacy - Digital tools proficiency',
    'TL2': 'Technology Literacy - Software applications',
    'TL3': 'Technology Literacy - Online platforms',
    'PSD1': 'Personal & Social Development - Self-awareness',
    'PSD2': 'Personal & Social Development - Emotional intelligence',
    'PSD3': 'Personal & Social Development - Social responsibility',
    'PSD4': 'Personal & Social Development - Cultural awareness',
    'PSD5': 'Personal & Social Development - Ethical behavior',
    'IM1': 'Information Management - Research skills',
    'IM2': 'Information Management - Data organization',
    'IM3': 'Information Management - Information evaluation',
    'IM4': 'Information Management - Digital literacy',
    'IM5': 'Information Management - Knowledge management',
    'IM6': 'Information Management - Information security',
    'W1': 'Work Readiness - Professional attitude',
    'W2': 'Work Readiness - Workplace behavior',
    'W3': 'Work Readiness - Career planning',
    'Employed': 'Previous employment status (0: No, 1: Yes)',
    'Score': 'Overall academic score (0-100)'
}

# Test data for demonstration
TEST_DATA = {
    'Gender': 1, 'Nationality': 0, 'Major': 1, 'Level': 3,
    'IE1': 4, 'IE2': 3, 'IE3': 4, 'IE4': 3, 'IE5': 4,
    'SMSK1': 4, 'SMSK2': 5, 'SMSK3': 3, 'SMSK4': 4,
    'RAS1': 4, 'RAS2': 3, 'RAS3': 3, 'RAS4': 4, 'RAS5': 3,
    'TL1': 5, 'TL2': 4, 'TL3': 5,
    'PSD1': 4, 'PSD2': 4, 'PSD3': 3, 'PSD4': 4, 'PSD5': 4,
    'IM1': 4, 'IM2': 4, 'IM3': 3, 'IM4': 5, 'IM5': 4, 'IM6': 4,
    'W1': 4, 'W2': 4, 'W3': 3,
    'Employed': 1, 'Score': 85
}

# Anavid branding banner
st.markdown("""
<div style="background: linear-gradient(90deg, #0066cc 0%, #0052a3 100%); 
            padding: 1.5rem; 
            border-radius: 8px; 
            text-align: center;
            margin-bottom: 2rem;">
    <a href="https://www.anavid.ai/fr/home" target="_blank" style="text-decoration: none;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">
            🎓 Employability Predictor
        </h1>
        <p style="color: #e6f2ff; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Powered by Anavid AI - Advanced Student Analytics
        </p>
    </a>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <a href="https://www.anavid.ai/fr/home" target="_blank">
            <img src="https://www.anavid.ai/img/logo.svg" width="150">
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Project Information")
    st.info("""
    **Anavid Assignment**
    
    Student Employability Prediction System using Machine Learning
    
    This application predicts the likelihood of student employment based on various academic and skill indicators.
    """)
    
    st.markdown("### 🤖 Model Information")
    st.success("""
    **Algorithm:** Random Forest Classifier
    
    **Features:** 37 input variables
    - Demographics (4)
    - Internship Experience (5)
    - Soft & Management Skills (4)
    - Research & Analytical Skills (5)
    - Technology Literacy (3)
    - Personal & Social Development (5)
    - Information Management (6)
    - Work Readiness (3)
    - Employment & Score (2)
    """)
    
    st.markdown("### 📖 Usage Guide")
    st.warning("""
    **Manual Entry:**
    1. Fill in all 37 fields
    2. Use Load Test Data for demo
    3. Click Predict to get results
    
    **Excel Upload:**
    1. Upload Excel file with all features
    2. Preview your data
    3. Get batch predictions
    4. Download results as CSV
    """)
    
    st.markdown("---")
    st.caption("© 2026 Anavid AI - All Rights Reserved")

# Main content
if model is None:
    st.error("⚠️ Model artifacts not found. Please ensure model.pkl, imputer.pkl, and feature_names.pkl are in the directory.")
    st.stop()

# Tab selection
tab1, tab2 = st.tabs(["📝 Manual Entry", "📊 Excel Upload"])

# TAB 1: Manual Entry
with tab1:
    st.markdown("### Enter Student Information")
    st.markdown("Fill in all 37 features below to predict employability:")
    
    # Initialize session state
    if 'input_values' not in st.session_state:
        st.session_state.input_values = {feature: 0 for feature in FEATURE_INFO.keys()}
    
    # Action buttons at top
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        if st.button("📥 Load Test Data", use_container_width=True):
            st.session_state.input_values = TEST_DATA.copy()
            st.rerun()
    with col_btn2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.input_values = {feature: 0 for feature in FEATURE_INFO.keys()}
            st.rerun()
    
    st.markdown("---")
    
    # Input fields in 3 columns
    features_list = list(FEATURE_INFO.keys())
    chunk_size = len(features_list) // 3 + (1 if len(features_list) % 3 else 0)
    
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    
    for idx, feature in enumerate(features_list):
        col_idx = idx // chunk_size
        if col_idx >= 3:
            col_idx = 2
        
        with columns[col_idx]:
            if feature in ['Gender', 'Nationality']:
                options = ['Female', 'Male'] if feature == 'Gender' else ['Omani', 'Non-Omani']
                value = st.selectbox(
                    f"{idx+1}. {feature}",
                    options=[0, 1],
                    format_func=lambda x: options[x],
                    key=f"input_{feature}",
                    index=int(st.session_state.input_values.get(feature, 0)),
                    help=FEATURE_INFO[feature]
                )
            elif feature == 'Major':
                options = ['Business', 'Engineering', 'IT', 'Science', 'Arts', 'Other']
                value = st.selectbox(
                    f"{idx+1}. {feature}",
                    options=list(range(6)),
                    format_func=lambda x: options[x],
                    key=f"input_{feature}",
                    index=int(st.session_state.input_values.get(feature, 0)),
                    help=FEATURE_INFO[feature]
                )
            elif feature == 'Level':
                options = ['First Year', 'Second Year', 'Third Year', 'Fourth Year', 'Graduate']
                value = st.selectbox(
                    f"{idx+1}. {feature}",
                    options=list(range(5)),
                    format_func=lambda x: options[x],
                    key=f"input_{feature}",
                    index=int(st.session_state.input_values.get(feature, 0)),
                    help=FEATURE_INFO[feature]
                )
            elif feature == 'Employed':
                options = ['No', 'Yes']
                value = st.selectbox(
                    f"{idx+1}. {feature}",
                    options=[0, 1],
                    format_func=lambda x: options[x],
                    key=f"input_{feature}",
                    index=int(st.session_state.input_values.get(feature, 0)),
                    help=FEATURE_INFO[feature]
                )
            elif feature == 'Score':
                value = st.number_input(
                    f"{idx+1}. {feature}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.input_values.get(feature, 0)),
                    step=1.0,
                    key=f"input_{feature}",
                    help=FEATURE_INFO[feature]
                )
            else:
                value = st.number_input(
                    f"{idx+1}. {feature}",
                    min_value=0,
                    max_value=5,
                    value=int(st.session_state.input_values.get(feature, 0)),
                    step=1,
                    key=f"input_{feature}",
                    help=FEATURE_INFO[feature]
                )
            
            st.session_state.input_values[feature] = value
    
    st.markdown("---")
    
    # Predict button
    col_predict = st.columns([2, 1, 2])[1]
    with col_predict:
        predict_button = st.button("🎯 PREDICT EMPLOYABILITY", use_container_width=True, type="primary")
    
    if predict_button:
        # Prepare input data
        input_data = pd.DataFrame([st.session_state.input_values])
        
        # Align with feature names
        if feature_names is not None:
            input_data = input_data[feature_names]
        
        # Handle missing values
        input_data_imputed = imputer.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_imputed)[0]
        prediction_proba = model.predict_proba(input_data_imputed)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        # Result banner
        if prediction == 1:
            st.success("### ✅ EMPLOYABLE - High likelihood of employment")
            result_color = "#28a745"
        else:
            st.error("### ❌ NOT EMPLOYABLE - Low likelihood of employment")
            result_color = "#dc3545"
        
        # Confidence scores
        st.markdown("### Confidence Scores")
        col_conf1, col_conf2 = st.columns(2)
        
        with col_conf1:
            st.metric(
                label="Not Employable Probability",
                value=f"{prediction_proba[0]*100:.2f}%"
            )
            st.progress(prediction_proba[0])
        
        with col_conf2:
            st.metric(
                label="Employable Probability",
                value=f"{prediction_proba[1]*100:.2f}%"
            )
            st.progress(prediction_proba[1])
        
        # Probability bar chart
        st.markdown("### Probability Distribution")
        prob_df = pd.DataFrame({
            'Category': ['Not Employable', 'Employable'],
            'Probability': [prediction_proba[0]*100, prediction_proba[1]*100]
        })
        
        st.bar_chart(prob_df.set_index('Category'))
        
        # Additional insights
        st.markdown("### 💡 Key Insights")
        confidence = max(prediction_proba) * 100
        
        if confidence >= 80:
            confidence_level = "Very High"
            confidence_color = "#28a745"
        elif confidence >= 60:
            confidence_level = "High"
            confidence_color = "#ffc107"
        elif confidence >= 50:
            confidence_level = "Moderate"
            confidence_color = "#fd7e14"
        else:
            confidence_level = "Low"
            confidence_color = "#dc3545"
        
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 5px solid {confidence_color};">
            <h4 style="margin-top: 0;">Prediction Confidence: {confidence_level} ({confidence:.1f}%)</h4>
            <p><strong>Overall Score:</strong> {st.session_state.input_values['Score']:.1f}/100</p>
            <p><strong>Employment History:</strong> {'Yes' if st.session_state.input_values['Employed'] == 1 else 'No'}</p>
            <p><strong>Academic Level:</strong> {['First Year', 'Second Year', 'Third Year', 'Fourth Year', 'Graduate'][st.session_state.input_values['Level']]}</p>
        </div>
        """, unsafe_allow_html=True)

# TAB 2: Excel Upload
with tab2:
    st.markdown("### Upload Excel File for Batch Predictions")
    
    st.info("""
    **Instructions:**
    1. Upload an Excel file (.xlsx) containing student data
    2. The file should include all 37 features as columns
    3. Each row represents one student
    4. The system will automatically predict employability for all students
    """)
    
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload Excel file with student data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df_upload = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df_upload)} students.")
            
            # Show preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df_upload.head(10), use_container_width=True)
            
            st.markdown(f"**Total Rows:** {len(df_upload)} | **Total Columns:** {len(df_upload.columns)}")
            
            # Predict button
            if st.button("🎯 Generate Predictions", use_container_width=True, type="primary"):
                with st.spinner("Generating predictions..."):
                    # Prepare data
                    df_predict = df_upload.copy()
                    
                    # Remove ID column if present
                    if 'ID' in df_predict.columns:
                        ids = df_predict['ID']
                        df_predict = df_predict.drop('ID', axis=1)
                    else:
                        ids = None
                    
                    # Align with feature names
                    if feature_names is not None:
                        # Ensure all required features are present
                        missing_features = set(feature_names) - set(df_predict.columns)
                        if missing_features:
                            st.error(f"Missing features: {missing_features}")
                            st.stop()
                        df_predict = df_predict[feature_names]
                    
                    # Handle missing values
                    df_imputed = imputer.transform(df_predict)
                    
                    # Make predictions
                    predictions = model.predict(df_imputed)
                    predictions_proba = model.predict_proba(df_imputed)
                    
                    # Create results dataframe
                    results_df = df_upload.copy()
                    results_df['Prediction'] = predictions
                    results_df['Prediction_Label'] = ['Employable' if p == 1 else 'Not Employable' for p in predictions]
                    results_df['Confidence_Not_Employable'] = predictions_proba[:, 0]
                    results_df['Confidence_Employable'] = predictions_proba[:, 1]
                    results_df['Confidence_Score'] = [max(proba) for proba in predictions_proba]
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Batch Prediction Results")
                    
                    # Summary statistics
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric("Total Students", len(predictions))
                    with col_stat2:
                        employable_count = sum(predictions == 1)
                        st.metric("Employable", employable_count)
                    with col_stat3:
                        not_employable_count = sum(predictions == 0)
                        st.metric("Not Employable", not_employable_count)
                    with col_stat4:
                        avg_confidence = results_df['Confidence_Score'].mean() * 100
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    
                    # Distribution chart
                    st.markdown("### Prediction Distribution")
                    dist_df = pd.DataFrame({
                        'Category': ['Not Employable', 'Employable'],
                        'Count': [not_employable_count, employable_count]
                    })
                    st.bar_chart(dist_df.set_index('Category'))
                    
                    # Confusion matrix visualization (if ground truth available)
                    if 'Actual' in results_df.columns or 'Target' in results_df.columns:
                        st.markdown("### Confusion Matrix")
                        actual_col = 'Actual' if 'Actual' in results_df.columns else 'Target'
                        
                        try:
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            from sklearn.metrics import confusion_matrix
                            
                            cm = confusion_matrix(results_df[actual_col], predictions)
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                       xticklabels=['Not Employable', 'Employable'],
                                       yticklabels=['Not Employable', 'Employable'])
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)
                        except ImportError:
                            st.warning("Matplotlib/Seaborn not available for confusion matrix visualization")
                    
                    # Show detailed results
                    st.markdown("### Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    st.markdown("### 💾 Download Results")
                    
                    # Convert to CSV
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv_data,
                        file_name="employability_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your Excel file has the correct format with all 37 features.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p style="margin: 0;">
        <strong>Employability Predictor</strong> - Anavid Assignment
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        Powered by <a href="https://www.anavid.ai/fr/home" target="_blank" style="color: #0066cc; text-decoration: none;">Anavid AI</a> | 
        Advanced Machine Learning for Student Analytics
    </p>
</div>
""", unsafe_allow_html=True)
