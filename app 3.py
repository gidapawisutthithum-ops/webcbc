import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import io

# --- 1. Page Configuration ---
st.set_page_config(page_title="Hematology Clinical Decision Support", layout="wide")

# --- 2. Model Training (Cached for performance) ---
@st.cache_resource
def load_and_train_model():
    # Load dataset
    try:
        df = pd.read_csv("final_combined_training_data.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'final_combined_training_data.csv' is in the same directory.")
        return None, None, None
    
    # Define features and target
    features = ['WBC', 'LYMp', 'NEUTp', 'LYMn', 'NEUTn', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'PDW', 'PCT']
    X = df[features].fillna(df[features].median())
    y = df['Diagnosis']
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    
    return model, le, features

model, le, feature_cols = load_and_train_model()

# --- 3. Helper Functions ---
def calculate_indices(mcv, rbc, rdw=None):
    mentzer_index = None
    rdw_index = None
    if mcv and rbc and rbc > 0:
        mentzer_index = mcv / rbc
        if rdw and rdw > 0:
            rdw_index = mentzer_index * rdw
    return mentzer_index, rdw_index

def evaluate_key_findings(data, mentzer_index):
    findings = []
    if mentzer_index:
        if mentzer_index < 13:
            findings.append("Low Mentzer Index (<13) suggests possible Thalassemia Trait.")
        else:
            findings.append("High Mentzer Index (>13) suggests possible Iron Deficiency Anemia.")
    
    if data.get('MCV', 0) < 80:
        findings.append("Low MCV detected (Microcytosis).")
    elif data.get('MCV', 0) > 100:
        findings.append("High MCV detected (Macrocytosis).")
        
    if data.get('HGB', 0) < 12: # Approximate general threshold
        findings.append("Low Hemoglobin detected (Anemia).")
        
    return findings

def get_recommendations(prediction, findings):
    recs = ["Complete Blood Count (CBC) follow-up in 3-6 months."]
    if "Thalassemia" in str(prediction) or any("Thalassemia" in f for f in findings):
        recs.append("Recommend Hemoglobin Typing / Hb Electrophoresis.")
        recs.append("Consult Genetic Counselor if planning a family.")
    if "Iron deficiency" in str(prediction) or any("Iron Deficiency" in f for f in findings):
        recs.append("Check Serum Ferritin, TIBC, and Serum Iron.")
    if "Leukemia" in str(prediction):
        recs.append("URGENT: Refer to Hematologist for peripheral blood smear and possible bone marrow examination.")
    return recs

def calculate_confidence(input_data):
    # Mock confidence score based on missing values and extreme outliers
    missing = sum(1 for v in input_data.values() if v == 0 or pd.isna(v))
    total = len(input_data)
    completeness = ((total - missing) / total) * 100
    
    # Decrease confidence slightly if extreme values are found (sanity check)
    if input_data.get('WBC', 0) > 100 or input_data.get('HGB', 0) > 25:
        completeness -= 15
        
    return max(min(completeness, 99.0), 10.0) # Cap between 10% and 99%

def predict_disease(input_data):
    if not model:
        return "Model not loaded", 0.0, []
    
    df_input = pd.DataFrame([input_data])
    # Ensure correct columns
    for col in feature_cols:
        if col not in df_input:
            df_input[col] = 0.0
            
    df_input = df_input[feature_cols]
    
    # Predict
    prob = model.predict_proba(df_input)[0]
    pred_idx = np.argmax(prob)
    prediction = le.inverse_transform([pred_idx])[0]
    probability = prob[pred_idx] * 100
    
    return prediction, probability, prob

def calculate_child_probability(father_pred, mother_pred):
    # Simplified Mendelian genetics logic for demonstration (assuming trait carriers)
    thal_keywords = ["Thalassemia", "Microcytic"]
    
    father_risk = any(k in father_pred for k in thal_keywords)
    mother_risk = any(k in mother_pred for k in thal_keywords)
    
    if father_risk and mother_risk:
        return "25% High Risk (Major), 50% Carrier Trait, 25% Unaffected"
    elif father_risk or mother_risk:
        return "0% High Risk (Major), 50% Carrier Trait, 50% Unaffected"
    else:
        return "Low risk of inheriting significant blood disorders based on current data."

# --- 4. User Interface ---
st.title("🩸 Hematology & Family Planning Clinical Decision Support System")

mode = st.sidebar.radio("Select Analysis Mode:", ["Mode A: Individual Analysis", "Mode B: Family Planning"])

def data_input_form(person_label):
    st.subheader(f"{person_label} Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input(f"Age ({person_label})", min_value=0, max_value=120, value=30, key=f"age_{person_label}")
    with col2:
        gender = st.selectbox(f"Gender ({person_label})", ["Male", "Female"], key=f"gender_{person_label}")
    with col3:
        fam_history = st.selectbox(f"Family History of Blood Disorders", ["Unknown", "Yes", "No"], key=f"fam_{person_label}")

    st.markdown("**CBC Parameters**")
    
    # File Upload Option
    uploaded_file = st.file_uploader(f"Upload CSV to auto-populate ({person_label})", type=["csv"], key=f"file_{person_label}")
    
    input_vals = {}
    defaults = {col: 0.0 for col in feature_cols}
    defaults['RDW'] = 0.0
    
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            if not user_df.empty:
                for col in defaults.keys():
                    if col in user_df.columns:
                        defaults[col] = float(user_df.iloc[0][col])
        except Exception as e:
            st.warning(f"Could not read file: {e}")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        input_vals['WBC'] = st.number_input("WBC", value=defaults.get('WBC', 0.0), key=f"wbc_{person_label}")
        input_vals['LYMp'] = st.number_input("LYMp (%)", value=defaults.get('LYMp', 0.0), key=f"lymp_{person_label}")
        input_vals['NEUTp'] = st.number_input("NEUTp (%)", value=defaults.get('NEUTp', 0.0), key=f"neutp_{person_label}")
        input_vals['LYMn'] = st.number_input("LYMn", value=defaults.get('LYMn', 0.0), key=f"lymn_{person_label}")
    with col_b:
        input_vals['NEUTn'] = st.number_input("NEUTn", value=defaults.get('NEUTn', 0.0), key=f"neutn_{person_label}")
        input_vals['RBC'] = st.number_input("RBC", value=defaults.get('RBC', 0.0), key=f"rbc_{person_label}")
        input_vals['HGB'] = st.number_input("HGB", value=defaults.get('HGB', 0.0), key=f"hgb_{person_label}")
        input_vals['HCT'] = st.number_input("HCT", value=defaults.get('HCT', 0.0), key=f"hct_{person_label}")
    with col_c:
        input_vals['MCV'] = st.number_input("MCV", value=defaults.get('MCV', 0.0), key=f"mcv_{person_label}")
        input_vals['MCH'] = st.number_input("MCH", value=defaults.get('MCH', 0.0), key=f"mch_{person_label}")
        input_vals['MCHC'] = st.number_input("MCHC", value=defaults.get('MCHC', 0.0), key=f"mchc_{person_label}")
        input_vals['PLT'] = st.number_input("PLT", value=defaults.get('PLT', 0.0), key=f"plt_{person_label}")
    with col_d:
        input_vals['PDW'] = st.number_input("PDW", value=defaults.get('PDW', 0.0), key=f"pdw_{person_label}")
        input_vals['PCT'] = st.number_input("PCT", value=defaults.get('PCT', 0.0), key=f"pct_{person_label}")
        input_vals['RDW'] = st.number_input("RDW (Optional)", value=defaults.get('RDW', 0.0), key=f"rdw_{person_label}")

    return input_vals, age, gender, fam_history

def display_results(person_label, input_vals):
    st.markdown(f"### Results for {person_label}")
    
    # Predict
    pred_class, pred_prob, all_probs = predict_disease(input_vals)
    conf_score = calculate_confidence(input_vals)
    
    # Calculate Indices
    mentzer, rdw_idx = calculate_indices(input_vals['MCV'], input_vals['RBC'], input_vals['RDW'])
    
    # Findings & Recs
    findings = evaluate_key_findings(input_vals, mentzer)
    recs = get_recommendations(pred_class, findings)
    
    # Layout Results
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        st.success(f"**Primary Prediction:** {pred_class} ({pred_prob:.1f}%)")
        st.info(f"**Data Confidence Score:** {conf_score:.1f}%")
        st.write(f"**Mentzer Index:** {mentzer:.2f}" if mentzer else "**Mentzer Index:** N/A (Missing RBC/MCV)")
        if input_vals['RDW'] > 0:
            st.write(f"**RDW Index:** {rdw_idx:.2f}" if rdw_idx else "**RDW Index:** N/A")
            
    with r_col2:
        st.markdown("**Key Findings:**")
        if findings:
            for f in findings:
                st.write(f"- ⚠️ {f}")
        else:
            st.write("- No significant primary abnormalities detected in indices.")
            
        st.markdown("**Further Lab Recommendations:**")
        for r in recs:
            st.write(f"- 🔬 {r}")
            
    return pred_class

# --- App Execution ---
if mode == "Mode A: Individual Analysis":
    ind_data, age, gen, fam = data_input_form("Patient")
    
    if st.button("Run Clinical Analysis"):
        st.divider()
        display_results("Patient", ind_data)
        
        # Report generation string buffer
        report_text = f"CLINICAL REPORT - INDIVIDUAL\n\nAge: {age} | Gender: {gen} | Family History: {fam}\n\n"
        st.download_button(
            label="📄 Print / Export Report",
            data=report_text,
            file_name="clinical_report.txt",
            mime="text/plain"
        )

elif mode == "Mode B: Family Planning":
    st.info("Please enter CBC profiles for both Prospective Father and Prospective Mother.")
    
    tab1, tab2 = st.tabs(["Prospective Father", "Prospective Mother"])
    
    with tab1:
        dad_data, dad_age, dad_gen, dad_fam = data_input_form("Father")
    with tab2:
        mom_data, mom_age, mom_gen, mom_fam = data_input_form("Mother")
        
    if st.button("Run Family Planning Analysis"):
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            dad_pred = display_results("Prospective Father", dad_data)
        with col2:
            mom_pred = display_results("Prospective Mother", mom_data)
            
        st.divider()
        st.subheader("🧬 Child's Genetic Probability")
        genetic_result = calculate_child_probability(dad_pred, mom_pred)
        st.markdown(f"### **{genetic_result}**")
        st.caption("Note: This is a predictive heuristic based on CBC phenotypic expression and should be confirmed with clinical genetic testing.")