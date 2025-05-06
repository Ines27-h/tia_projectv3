import streamlit as st
import joblib
import numpy as np
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModel

# --- Laden van model en selector ---
model = joblib.load('tia_model.joblib')
selector = joblib.load('text_feature_selector.joblib')

# --- Laden van MedRoBERTa.nl ---
tokenizer = AutoTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")
embedding_model = AutoModel.from_pretrained("CLTL/MedRoBERTa.nl")

# --- Functies ---
@st.cache_resource
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def sanitize_column_name(name):
    name = re.sub(r'[\[\]<>]', '_', name)
    name = name.replace(' ', '_')
    return name

def create_input_vector(user_inputs, anamnese_embedding):
    feature_names = [f"anamnese_feature_mean_{i}" for i in range(len(anamnese_embedding))]
    text_features = pd.DataFrame([anamnese_embedding], columns=feature_names)
    selected_text_features = selector.transform(text_features)
    selected_text_df = pd.DataFrame(selected_text_features, columns=[
        col for col, keep in zip(text_features.columns, selector.get_support()) if keep
    ])
    dummy_data = user_inputs.copy()

    # Bloeddrukcategorie afleiden
    dummy_data['bp_category_normaal'] = 1 if dummy_data['systolic_imputed'] < 140 else 0
    dummy_data['bp_category_hoog'] = 1 if dummy_data['systolic_imputed'] >= 140 else 0

    # Leeftijdsgroep dynamisch bepalen
    age = dummy_data['age']

    dummy_data['age_group_<40'] = 0
    dummy_data['age_group_40-50'] = 0
    dummy_data['age_group_50-60'] = 0
    dummy_data['age_group_60-70'] = 0
    dummy_data['age_group_70-80'] = 0
    dummy_data['age_group_>80'] = 0

    if age < 40:
        dummy_data['age_group_<40'] = 1
    elif 40 <= age < 50:
        dummy_data['age_group_40-50'] = 1
    elif 50 <= age < 60:
        dummy_data['age_group_50-60'] = 1
    elif 60 <= age < 70:
        dummy_data['age_group_60-70'] = 1
    elif 70 <= age < 80:
        dummy_data['age_group_70-80'] = 1
    elif age >= 80:
        dummy_data['age_group_>80'] = 1

    dummy_df = pd.DataFrame([dummy_data])

    input_vector = pd.concat([dummy_df.reset_index(drop=True), selected_text_df.reset_index(drop=True)], axis=1)
    input_vector.columns = [sanitize_column_name(col) for col in input_vector.columns]
    return input_vector

# --- Streamlit UI ---
st.set_page_config(page_title="TIA Voorspelling", page_icon="🚑", layout="centered")
st.title("🚑 TIA Voorspellingstool voor Huisartsen")
st.markdown("Voer de klinische gegevens van de patiënt in om een voorspelling te maken.")

# Inputvelden
anamnese_input = st.text_area("Anamnese tekst", placeholder="Bijv. Patiënt had plots spraakproblemen en krachtverlies...")
age = st.number_input("Leeftijd (in jaren)", min_value=0, max_value=120, value=65)
seks = st.selectbox("Geslacht", ["Man", "Vrouw"])

systolic_bp = st.number_input("Bloeddruk (systolisch)", min_value=80, max_value=250, value=140)
diastolic_bp = st.number_input("Bloeddruk (diastolisch)", min_value=40, max_value=150, value=80)

# Risicofactoren
st.markdown("### Risicofactoren:")
hypertension = st.checkbox("Hypertensie")
atriumfibrillation = st.checkbox("Atriumfibrilleren")
hypercholesterolaemia = st.checkbox("Hypercholesterolemie")
diabetes = st.checkbox("Diabetes")
ischaemicstroke = st.checkbox("Eerder herseninfarct")
intracranialhaemorrhage = st.checkbox("Eerder hersenbloeding")
previoustia = st.checkbox("Eerdere TIA")
peripheralarterydisease = st.checkbox("Perifeer arterieel vaatlijden")
myocardialinfarction = st.checkbox("Hartinfarct")
smoking = st.checkbox("Roken")
alcoholabusus = st.checkbox("Alcoholabusus")
positivefamilyhistory = st.checkbox("Positieve familieanamnese")
adipositas = st.checkbox("Adipositas")

# Extra klinische gegevens
st.markdown("### Extra klinische gegevens:")

# Duur van de klachten
duur_klachten = st.selectbox(
    "Duur van de klachten:",
    ["<10 minuten", "10-59 minuten", ">60 minuten"]
)
duur_mapping = {"<10 minuten": 0, "10-59 minuten": 1, ">60 minuten": 2}
duur_klachten_encoded = duur_mapping[duur_klachten]

# Beoordeling lichamelijk onderzoek
beoordeling_lo = st.selectbox(
    "Beoordeling lichamelijk onderzoek:",
    ["Geen bijzonderheden", "Twijfel", "Afwijkend"]
)

if st.button("Voorspel TIA"):
    if anamnese_input.strip() == "":
        st.warning("Voer eerst een anamnese tekst in.")
    else:
        user_inputs = {
            'age': age,
            'seks': 1 if seks == "Man" else 0,
            'hypertension': int(hypertension),
            'atriumfibrillation': int(atriumfibrillation),
            'hypercholesterolaemia': int(hypercholesterolaemia),
            'diabetes': int(diabetes),
            'ischaemicstroke': int(ischaemicstroke),
            'intracranialhaemorrhage': int(intracranialhaemorrhage),
            'previoustia': int(previoustia),
            'peripheralarterydisease': int(peripheralarterydisease),
            'myocardialinfarction': int(myocardialinfarction),
            'smoking': int(smoking),
            'alcoholabusus': int(alcoholabusus),
            'positivefamilyhistory': int(positivefamilyhistory),
            'adipositas': int(adipositas),
            'systolic_imputed': systolic_bp,
            'diastolic_imputed': diastolic_bp,
            'gezond': 0,
            'risk_score': 0,
            'duur_klachten_encoded': duur_klachten_encoded,
            'afwijkend_neurologisch_onderzoek': 0,
            'beoordeling_lo_Geen_bijzonderheden': 1 if beoordeling_lo == "Geen bijzonderheden" else 0,
            'beoordeling_lo_Twijfel': 1 if beoordeling_lo == "Twijfel" else 0,
            'beoordeling_lo_Afwijkend': 1 if beoordeling_lo == "Afwijkend" else 0
        }

        # Embed de tekst
        embedding = embed_text(anamnese_input)

        # Maak inputvector
        input_vector = create_input_vector(user_inputs, embedding)

        # --- Hier reindex toevoegen ---
        feature_list = joblib.load('feature_list.joblib')
        input_vector = input_vector.reindex(columns=feature_list, fill_value=0)

        # Voorspel
        probability = model.predict_proba(input_vector)[0, 1]
        prediction = model.predict(input_vector)[0]

        st.subheader("Resultaat:")
        if prediction == 1:
            st.success(f"💥 Mogelijke TIA! (Voorspelkans: {probability:.2f})")
        else:
            st.info(f"✅ Geen TIA voorspeld. (Voorspelkans: {probability:.2f})")

        st.markdown("---")
        st.caption("Model: Logistic Regression + SelectKBest + MedRoBERTa.nl embedding")
