import os
import pickle
from unittest.util import _MAX_LENGTH
import streamlit as st
from streamlit_option_menu import option_menu
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from collections import Counter
import re
from auth import register_user, login_user, create_connection
from create_db import insert_patient_data, insert_diabetes_data, insert_heart_disease_data, insert_parkinsons_data, retrieve_patient_data, retrieve_parkinsons_data, retrieve_diabetes_data, retrieve_heart_disease_data
from fpdf import FPDF
import io
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="HealthPredictX",
                   layout="wide",
                   page_icon="🧑‍⚕️")

#device = 0 if torch.cuda.is_available() else -1

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def create_pdf(patient_data):
    # Initialize FPDF object
    pdf = FPDF()
    pdf.add_page()
    
    # Set font and title
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Comprehensive Medical Report", ln=True, align="C")
    pdf.ln(10)  # Line break
    
    # Add patient details from the dictionary
    pdf.cell(200, 10, txt=f"Name: {patient_data['Name']}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {patient_data['Age']}", ln=True)
    pdf.cell(200, 10, txt=f"Sex: {patient_data['Sex']}", ln=True)
    pdf.ln(5)  # Line break for better formatting
    
    # Add health assessment details
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Diabetes Assessment:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"  Verdict: {patient_data['Diabetes Verdict']}", ln=True)
    pdf.cell(200, 10, txt=f"  Risk: {patient_data['Risk of Diabetes']}", ln=True)
    pdf.ln(5)  # Line break
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Heart Disease Assessment:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"  Verdict: {patient_data['Heart Disease Verdict']}", ln=True)
    pdf.cell(200, 10, txt=f"  Risk: {patient_data['Risk of Heart Disease']}", ln=True)
    pdf.ln(5)  # Line break
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Parkinson's Disease Assessment:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"  Verdict: {patient_data['Parkinsons Verdict']}", ln=True)
    pdf.cell(200, 10, txt=f"  Risk: {patient_data['Risk of Parkinsons']}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="This report is based on preliminary health assessments and may require further clinical evaluation.")
    
    pdf_buffer = BytesIO()
    
    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_buffer.write(pdf_output)
    
    pdf_buffer.seek(0)

    return pdf_buffer

def register():
    st.title("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Register"):
        connection = create_connection()
        cursor = connection.cursor()
        # Check if username already exists in the database
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cursor.fetchone():  # If the username is already taken
            st.warning("Username already exists!")
        else:
            register_user(username, password)
            st.success("User registered successfully! You can now log in.")
        connection.close()

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Login"):
        if login_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.warning("Invalid username or password.")

if st.session_state['logged_in']:
    suggestion_generator = pipeline('text-generation', model='gpt2', device=0, max_length=1000)

    def clean_generated_text(text):
        # Step 1: Remove extra whitespaces and standardize punctuation
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation

        # Step 2: Split text into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)

        # Step 3: Filter out repeated sentences
        sentence_count = Counter(sentences)
        unique_sentences = [s for s in sentence_count if sentence_count[s] == 1]
        
        # Step 4: Join the unique sentences back into a single text block
        cleaned_text = ' '.join(unique_sentences)
        
        return cleaned_text


    def get_diabetes_health_suggestions(prediction, input_data):
        # Analyze potential risk factors
        prompt_1 = (
            f"Given the patient's health metrics:\n"
            f"- Pregnancies: {input_data[0]}\n"
            f"- Glucose Level: {input_data[1]} mg/dL\n"
            f"- Blood Pressure: {input_data[2]} mm Hg\n"
            f"- Skin Thickness: {input_data[3]} mm\n"
            f"- Insulin Level: {input_data[4]} IU/mL\n"
            f"- BMI: {input_data[5]}\n"
            f"- Diabetes Pedigree Function: {input_data[6]}\n"
            f"- Age: {input_data[7]} years\n\n"
            f"The diabetes prediction suggests that the patient {'is diabetic' if prediction == 1 else 'is not diabetic'}.\n"
            f"Please analyze the health metrics and identify potential risk factors:\n"
            f"- Use bullet points for clarity."
        )

        suggestions_1 = suggestion_generator(prompt_1, max_length=300, num_return_sequences=1, temperature=0.7)
        formatted_suggestions_1 = clean_generated_text(suggestions_1[0]['generated_text'].strip())

        # Health recommendations
        prompt_2 = (
            f"Given the patient's health metrics:\n"
            f"- Pregnancies: {input_data[0]}\n"
            f"- Glucose Level: {input_data[1]} mg/dL\n"
            f"- Blood Pressure: {input_data[2]} mm Hg\n"
            f"- Skin Thickness: {input_data[3]} mm\n"
            f"- Insulin Level: {input_data[4]} IU/mL\n"
            f"- BMI: {input_data[5]}\n"
            f"- Diabetes Pedigree Function: {input_data[6]}\n"
            f"- Age: {input_data[7]} years\n\n"
            f"The diabetes prediction suggests that the patient {'is diabetic' if prediction == 1 else 'is not diabetic'}.\n"
            f"Please provide health recommendations in the following format:\n"
            f"1. **Diagnostic Recommendations**: Suggest up to two tests.\n"
            f"2. **Dietary Recommendations**: Recommend two dietary changes.\n"
            f"3. **Exercise Plan**: Suggest one activity.\n"
            f"4. **Medications**: Suggest up to two medications.\n"
            f"5. **Lifestyle Modifications**: Recommend one change.\n"
            f"6. **Follow-Up Plan**: Suggest follow-up activities.\n"
            f"7. **Pregnancy Considerations**: Suggest one supplement if applicable.\n"
            f"8. **Breastfeeding Considerations**: Suggest one dietary change.\n"
            f"9. **Dietary Supplementation**: Recommend one supplement.\n"
            f"Use bullet points for clarity."
        )

        suggestions_2 = suggestion_generator(prompt_2, max_length=500, num_return_sequences=1, temperature=0.7)
        formatted_suggestions_2 = clean_generated_text(suggestions_2[0]['generated_text'].strip())
        
        return formatted_suggestions_1 + "\n\n" + formatted_suggestions_2





    def get_heart_disease_suggestions(prediction, input_data):

        prompt_1 = (
            f"Patient's health metrics:\n"
            f"- Age: {input_data[0]} years\n"
            f"- Anaemia: {'Yes' if input_data[1] == 1 else 'No'}\n"
            f"- Creatinine Phosphokinase: {input_data[2]} IU/L\n"
            f"- Diabetes: {'Yes' if input_data[3] == 1 else 'No'}\n"
            f"- Ejection Fraction: {input_data[4]}%\n"
            f"- High Blood Pressure: {'Yes' if input_data[5] == 1 else 'No'}\n"
            f"- Platelets: {input_data[6]} x10^9/L\n"
            f"- Serum Creatinine: {input_data[7]} mg/dL\n"
            f"- Serum Sodium: {input_data[8]} mEq/L\n"
            f"- Sex: {'Male' if input_data[9] == 1 else 'Female'}\n"
            f"- Smoking: {'Yes' if input_data[10] == 1 else 'No'}\n"
            f"- Time: {input_data[11]} months since diagnosis\n\n"
            f"The heart disease prediction suggests the patient has {'heart disease' if prediction == 1 else 'no heart disease'}.\n\n"
            f"Analyze the given health metrics, and identify potential {'contributing factors towards the diagnosis' if prediction == 1 else 'risk factors towards a future diagnosis'}.\n"
            f"Please provide a concise answer.\n"
        )

        prompt_2 = (
            f"Patient's health metrics:\n"
            f"- Age: {input_data[0]} years\n"
            f"- Anaemia: {'Yes' if input_data[1] == 1 else 'No'}\n"
            f"- Creatinine Phosphokinase: {input_data[2]} IU/L\n"
            f"- Diabetes: {'Yes' if input_data[3] == 1 else 'No'}\n"
            f"- Ejection Fraction: {input_data[4]}%\n"
            f"- High Blood Pressure: {'Yes' if input_data[5] == 1 else 'No'}\n"
            f"- Platelets: {input_data[6]} x10^9/L\n"
            f"- Serum Creatinine: {input_data[7]} mg/dL\n"
            f"- Serum Sodium: {input_data[8]} mEq/L\n"
            f"- Sex: {'Male' if input_data[9] == 1 else 'Female'}\n"
            f"- Smoking: {'Yes' if input_data[10] == 1 else 'No'}\n"
            f"- Time: {input_data[11]} months since diagnosis\n\n"
            f"The heart disease prediction suggests the patient has {'heart disease' if prediction == 1 else 'no heart disease'}.\n\n"
            f"Provide a comprehensive course of action, limited to specific and actionable recommendations for the clinician:\n\n"
            f"1. **Diagnostic Recommendations**: Suggest one or two specific tests to confirm the diagnosis or assess risk factors.\n"
            f"2. **Treatment Plan**:\n"
            f"   - **Dietary Interventions**: Provide up to two dietary changes personalized to the patient's condition.\n"
            f"   - **Exercise Prescriptions**: Recommend one exercise plan based on the patient's profile.\n"
            f"   - **Medication**: Mention any medications, if necessary.\n"
            f"3. **Lifestyle Modifications**: Offer one suggestion to improve lifestyle (e.g., stress management or sleep improvements).\n"
            f"4. **Follow-Up Plan**: Recommend one or two follow-up activities for monitoring the patient's heart health.\n\n"
            f"All recommendations should be personalized, evidence-based, and aligned with current medical guidelines."
        )

        # Generate suggestions with a max length to prevent unnecessary repetition
        suggestions_1 = suggestion_generator(prompt_1, max_length=500, num_return_sequences=1, temperature=0.7)

        # Clean and format the generated text
        raw_text = suggestions_1[0]['generated_text'].strip()
        formatted_suggestions = clean_generated_text(raw_text)

        suggestions_2 = suggestion_generator(prompt_2, max_length=500, num_return_sequences=1, temperature=0.7)

        raw_text1 = suggestions_2[0]['generated_text'].strip()
        formatted_suggestions1 = clean_generated_text(raw_text1)

        return formatted_suggestions+"\n"+formatted_suggestions1

    def get_parkinsons_health_suggestions(prediction, input_data):

        prompt_1 = (
            f"Patient's health metrics for Parkinson's prediction:\n"
            f"- MDVP:Fo(Hz): {input_data[0]}\n"
            f"- MDVP:Fhi(Hz): {input_data[1]}\n"
            f"- MDVP:Flo(Hz): {input_data[2]}\n"
            f"- MDVP:Jitter(%): {input_data[3]}\n"
            f"- MDVP:Jitter(Abs): {input_data[4]}\n"
            f"- MDVP:RAP: {input_data[5]}\n"
            f"- MDVP:PPQ: {input_data[6]}\n"
            f"- Jitter:DDP: {input_data[7]}\n"
            f"- MDVP:Shimmer: {input_data[8]}\n"
            f"- MDVP:Shimmer(dB): {input_data[9]}\n"
            f"- Shimmer:APQ3: {input_data[10]}\n"
            f"- Shimmer:APQ5: {input_data[11]}\n"
            f"- MDVP:APQ: {input_data[12]}\n"
            f"- Shimmer:DDA: {input_data[13]}\n"
            f"- NHR: {input_data[14]}\n"
            f"- HNR: {input_data[15]}\n"
            f"- RPDE: {input_data[16]}\n"
            f"- DFA: {input_data[17]}\n"
            f"- Spread1: {input_data[18]}\n"
            f"- Spread2: {input_data[19]}\n"
            f"- D2: {input_data[20]}\n"
            f"- PPE: {input_data[21]}\n\n"
            f"The Parkinson's prediction suggests the patient has {'Parkinsons' if prediction == 1 else 'no Parkinsons'}.\n\n"
            f"Analyze the given health metrics, and identify potential {'contributing factors towards the diagnosis' if prediction == 1 else 'risk factors towards a future diagnosis'}.\n"
            f"Please provide a concise answer.\n"
        )

        prompt_2 = (
            f"Patient's health metrics for Parkinson's prediction:\n"
            f"- MDVP:Fo(Hz): {input_data[0]}\n"
            f"- MDVP:Fhi(Hz): {input_data[1]}\n"
            f"- MDVP:Flo(Hz): {input_data[2]}\n"
            f"- MDVP:Jitter(%): {input_data[3]}\n"
            f"- MDVP:Jitter(Abs): {input_data[4]}\n"
            f"- MDVP:RAP: {input_data[5]}\n"
            f"- MDVP:PPQ: {input_data[6]}\n"
            f"- Jitter:DDP: {input_data[7]}\n"
            f"- MDVP:Shimmer: {input_data[8]}\n"
            f"- MDVP:Shimmer(dB): {input_data[9]}\n"
            f"- Shimmer:APQ3: {input_data[10]}\n"
            f"- Shimmer:APQ5: {input_data[11]}\n"
            f"- MDVP:APQ: {input_data[12]}\n"
            f"- Shimmer:DDA: {input_data[13]}\n"
            f"- NHR: {input_data[14]}\n"
            f"- HNR: {input_data[15]}\n"
            f"- RPDE: {input_data[16]}\n"
            f"- DFA: {input_data[17]}\n"
            f"- Spread1: {input_data[18]}\n"
            f"- Spread2: {input_data[19]}\n"
            f"- D2: {input_data[20]}\n"
            f"- PPE: {input_data[21]}\n\n"
            f"The Parkinson's prediction suggests the patient has {'Parkinsons' if prediction == 1 else 'no Parkinsons'}.\n\n"
            f"Provide a concise course of action for the clinician, including:\n"
            f"1. **Diagnostic Recommendations**: Suggest 1-2 key additional tests or assessments that might be necessary.\n"
            f"2. **Treatment Plan**:\n"
            f"   - **Dietary Interventions**: Recommend up to two specific dietary changes tailored to the patient's profile.\n"
            f"   - **Exercise Prescriptions**: Suggest 1 personalized exercise plan, including the type and duration of exercises.\n"
            f"   - **Medications**: If necessary, mention relevant medications.\n"
            f"3. **Lifestyle Modifications**: Provide 1-2 recommendations for lifestyle improvements, such as stress reduction or sleep hygiene.\n"
            f"4. **Follow-Up Plan**: Suggest 1-2 specific follow-up steps to monitor the patient's condition.\n"
            f"5. **Consultation Note**: Remind the clinician to discuss the recommendations and customize them to the patient's needs.\n\n"
            f"All suggestions should be evidence-based and aligned with current medical guidelines.\n"
        )

        # Generate suggestions with appropriate constraints on length and creativity
        suggestions = suggestion_generator(prompt_1, max_length=500, num_return_sequences=1, temperature=0.7)
        raw_text = suggestions[0]['generated_text'].strip()
        formatted_suggestions = clean_generated_text(raw_text)

        suggestions_1 = suggestion_generator(prompt_2, max_length=500, num_return_sequences=1, temperature=0.7)
        raw_text1 = suggestions_1[0]['generated_text'].strip()
        formatted_suggestions1 = clean_generated_text(raw_text1)

        return formatted_suggestions+"\n"+formatted_suggestions1   

        
    # getting the working directory of the main.py
    working_dir = os.path.dirname(os.path.abspath(__file__))

    # loading the saved models

    #diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
    diabetes_model = pickle.load(open(f'{working_dir}/svc_diabetes.sav','rb'))

    #heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
    heart_disease_model = pickle.load(open(f'{working_dir}/logistic_model_updated.sav', 'rb'))

    parkinsons_model = pickle.load(open(f'{working_dir}/rf_model_updated.sav', 'rb'))

    # sidebar for navigation
    with st.sidebar:
        selected = option_menu('HealthPredictX',

                            ['Patient Data','Disease Predictions','Health Chatbot'],
                                #'Heart Disease',
                                #'Parkinsons'],
                            menu_icon='hospital-fill',
                            icons=['pen','activity','robot'],
                            default_index=0)

    if selected == 'Patient Data':
        st.title('Patient Data')

        choice = st.selectbox("Select an option",["Input Patient Details","Retrieve Patient Details","Diabetes","Heart Disease","Parkinson's"])

        if(choice == "Input Patient Details"):
            Name = ''
            Age = ''
            Gender = ''
            Address = ''
            Phone = ''
            Email = ''

            col1, col2, col3 = st.columns(3)

            with col1:
                Name = st.text_input("Name of the Patient")
            with col2:
                Age = st.text_input("Age of the Patient")
            with col3:
                Gender = st.text_input("Gender")
            with col1:
                Address = st.text_input("Address")
            with col2:
                Phone = st.text_input("Phone Number")
            with col3:
                Email = st.text_input("Email Address")

            if(st.button("Submit")):
                if Name and Age and Gender and Address and Phone and Email:
                    result = insert_patient_data(Name, Age, Gender, Address, Phone, Email)
                    if result:
                        st.success("Patient Record has been succesfully stored in the database!")
                else:
                    st.error("Error! Please ensure all fields have been filled in.")
        
        elif(choice == "Retrieve Patient Details"):
            Patient_Id = ''
            Name = ''
            Age = ''
            Gender = ''
            Address = ''
            Phone = ''
            Email = ''

            search_name = st.text_input("Enter the name of the patient")
            if(st.button("Submit")):
                if search_name:
                    result = retrieve_patient_data(search_name)
                    if result:
                        Patient_Id = result[0]
                        Name = result[1]
                        Age = result[2]
                        Gender = result[3]
                        Address = result[4]
                        Phone = result[5]
                        Email = result[6]
                        st.success(f"""
                        **Patient Details:**
                        - **Patient ID**: {result[0]}
                        - **Name**: {result[1]}
                        - **Age**: {result[2]}
                        - **Gender**: {result[3]}
                        - **Address**: {result[4]}
                        - **Phone**: {result[5]}
                        - **Email**: {result[6]}
                        """)
                else:
                    st.error("Error! Please ensure all fields have been filled in.")

        
        elif(choice == "Diabetes"):
            col1, col2, col3 = st.columns(3)

            patient_id = '' 
            pregnancies = '0'
            glucose = ''
            blood_pressure = ''
            skin_thickness = ''
            insulin = ''
            bmi = ''
            diabetes_pedigree = ''

            with col1:
                patient_id = st.text_input('Patient ID')

            with col2:
                glucose = st.text_input('Glucose Level')

            with col3:
                blood_pressure = st.text_input('Blood Pressure value')

            with col1:
                skin_thickness = st.text_input('Skin Thickness value')

            with col2:
                insulin = st.text_input('Insulin Level')

            with col3:
                bmi = st.text_input('BMI value')

            with col1:
                diabetes_pedigree = st.text_input('Diabetes Pedigree Function value')

            if(st.button("Submit")):
                if patient_id and pregnancies and glucose and blood_pressure and skin_thickness and insulin and bmi and diabetes_pedigree:
                    result = insert_diabetes_data(patient_id, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree)
                    if result:
                        st.success("Diabetes Data has been succesfully stored in the database!")
                else:
                    st.error("Error! Please ensure all fields have been filled in.")

        elif(choice == "Heart Disease"):
            col1, col2, col3 = st.columns(3)
            an = ''
            cp = ''
            db = ''
            ef = ''
            hbp = ''
            plt = ''
            sc = ''
            ss = ''
            sex = ''
            smk = ''
            ti = ''
            patient_id = ''

            with col1:
                patient_id = st.text_input('Patient ID')

            with col2:
                an = st.selectbox("Does the patient have anaemia?", ['Yes','No'])

            with col3:
                cp = st.text_input('Creatinine Phosphokinase')

            with col1:
                db = st.selectbox('Does the patient have diabetes?', ['Yes','No'])

            with col2:
                ef = st.text_input('Ejection Fraction')

            with col3:
                hbp = st.selectbox('Does the patient have high blood pressure?', ['Yes','No'])

            with col1:
                plt = st.text_input('Platelets')

            with col2:
                sc = st.text_input('Serum Creatinine')

            with col3:
                ss = st.text_input('Serum Sodium')

            with col1:
                smk = st.selectbox('Does the patient smoke?',['Yes','No'])

            with col2:
                ti = st.text_input('Follow up period')

            an_val = '1' if an=='Yes' else '0'
            db_val = '1' if db=='Yes' else '0'
            hbp_val = '1' if hbp=='Yes' else '0'
            smk_val = '1' if smk=='Yes' else '0'


            if(st.button("Submit")):
                if patient_id and an_val and cp and db_val and ef and hbp_val and plt and sc and ss and smk_val and ti:
                    result = insert_heart_disease_data(patient_id, an_val, cp, db_val, ef, hbp_val, plt, sc, ss, smk_val, ti)
                    if result:
                        st.success("Heart Disease data successfully stored in the database!")
                else:
                    st.error("Please ensure all fields have been filled in.")
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            patient_id = ''
            fo = ''
            fhi = ''
            flo = ''
            Jitter_percent = ''
            Jitter_Abs = ''
            RAP = ''
            PPQ = ''
            DDP = ''
            Shimmer = ''
            Shimmer_dB = ''
            APQ3 = ''
            APQ5 = ''
            APQ = ''
            DDA = ''
            NHR = ''
            HNR = ''
            RPDE = ''
            DFA = ''
            spread1 = ''
            spread2 = ''
            D2 = ''
            PPE = ''

            with col1:
                fo = st.text_input('MDVP:Fo(Hz)')

            with col2:
                fhi = st.text_input('MDVP:Fhi(Hz)')

            with col3:
                flo = st.text_input('MDVP:Flo(Hz)')

            with col4:
                Jitter_percent = st.text_input('MDVP:Jitter(%)')

            with col5:
                Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

            with col1:
                RAP = st.text_input('MDVP:RAP')

            with col2:
                PPQ = st.text_input('MDVP:PPQ')

            with col3:
                DDP = st.text_input('Jitter:DDP')

            with col4:
                Shimmer = st.text_input('MDVP:Shimmer')

            with col5:
                Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

            with col1:
                APQ3 = st.text_input('Shimmer:APQ3')

            with col2:
                APQ5 = st.text_input('Shimmer:APQ5')

            with col3:
                APQ = st.text_input('MDVP:APQ')

            with col4:
                DDA = st.text_input('Shimmer:DDA')

            with col5:
                NHR = st.text_input('NHR')

            with col1:
                HNR = st.text_input('HNR')

            with col2:
                RPDE = st.text_input('RPDE')

            with col3:
                DFA = st.text_input('DFA')

            with col4:
                spread1 = st.text_input('spread1')

            with col5:
                spread2 = st.text_input('spread2')

            with col1:
                D2 = st.text_input('D2')

            with col2:
                PPE = st.text_input('PPE')
            
            with col3:
                patient_id = st.text_input('Patient ID')

            if st.button("Submit"):
                if patient_id and fo and fhi and flo and Jitter_percent and Jitter_Abs and RAP and PPQ and DDP and Shimmer and Shimmer_dB and APQ3 and APQ5 and APQ and DDA and NHR and HNR and RPDE and DFA and spread1 and spread2 and D2 and PPE:
                    result = insert_parkinsons_data(patient_id, fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE)
                    if result:
                        st.success("Parkisons data successfully stored in the database!")
                else:
                    st.error("Please ensure all fields have been filled in.")
    
    if selected == 'Disease Predictions':
        st.title('Disease Predictions')
        
        # Input for patient ID and name
        patient_id = st.text_input("Enter the Patient ID")
        name = st.text_input("Enter the name of the patient")
        
        # Submit button for predictions
        if st.button("Submit"):
            if patient_id:
                # Retrieve patient and health data
                patient_result = retrieve_patient_data(name)
                age = patient_result[2]
                sex = '1' if patient_result[3] == 'Female' else '0'
                diabetes_result = retrieve_diabetes_data(patient_id)
                heart_result = retrieve_heart_disease_data(patient_id)
                parkinson_result = retrieve_parkinsons_data(patient_id)
                
                # Process diabetes prediction
                diabetes_input = [diabetes_result[2], diabetes_result[3], diabetes_result[4], diabetes_result[5], diabetes_result[6], diabetes_result[7], diabetes_result[8], age]
                diabetes_input = [float(x) for x in diabetes_input]
                diab_prediction = diabetes_model.predict([diabetes_input])
                diab_prob = diabetes_model.predict_proba([diabetes_input])

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The patient is diabetic'
                else:
                    diab_diagnosis = 'The patient is not diabetic'

                risk_score_d = diab_prob[0][1]
                risk_category_d = 'Low Risk' if risk_score_d < 0.3 else 'Medium Risk' if 0.3 <= risk_score_d < 0.7 else 'High Risk'
                st.success(diab_diagnosis)
                st.write(f'Risk of developing diabetes: {risk_score_d:.2f} ({risk_category_d})')

                # Process heart disease prediction
                heart_input = [age, heart_result[2], heart_result[3], heart_result[4], heart_result[5], heart_result[6], heart_result[7], heart_result[8], heart_result[9], sex, heart_result[10], heart_result[11]]
                heart_input = [float(x) for x in heart_input]
                heart_prediction = heart_disease_model.predict([heart_input])
                heart_prob = heart_disease_model.predict_proba([heart_input])

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The patient is having heart disease and is at risk of heart failure'
                else:
                    heart_diagnosis = 'The patient does not have any heart disease'
                    
                risk_score_h = heart_prob[0][1]
                risk_category_h = 'Low Risk' if risk_score_h < 0.3 else 'Medium Risk' if 0.3 <= risk_score_h < 0.7 else 'High Risk'
                st.success(heart_diagnosis)
                st.write(f'Risk of developing heart disease: {risk_score_h:.2f} ({risk_category_h})')

                # Process Parkinson's prediction
                parkinsons_input = [float(x) for x in parkinson_result[2:24]]  # Assuming indices 2-23 are relevant
                parkinson_prediction = parkinsons_model.predict([parkinsons_input])
                parkinson_prob = parkinsons_model.predict_proba([parkinsons_input])

                if parkinson_prediction[0] == 1:
                    parkinson_diagnosis = "The patient has Parkinson's disease"
                else:
                    parkinson_diagnosis = "The patient doesn't have Parkinson's disease"
                    
                risk_score_p = parkinson_prob[0][1]
                risk_category_p = 'Low Risk' if risk_score_p < 0.3 else 'Medium Risk' if 0.3 <= risk_score_p < 0.7 else 'High Risk'
                st.success(parkinson_diagnosis)
                st.write(f'Risk of developing Parkinsons: {risk_score_p:.2f} ({risk_category_p})')

                # Store the result in session state
                st.session_state['patient_data'] = {
                    'Name': patient_result[1],
                    'Age': age,
                    'Sex': patient_result[3],
                    'Diabetes Verdict': diab_diagnosis,
                    'Risk of Diabetes': f'Risk of developing diabetes: {risk_score_d:.2f} ({risk_category_d})',
                    'Heart Disease Verdict': heart_diagnosis,
                    'Risk of Heart Disease': f'Risk of developing heart disease: {risk_score_h:.2f} ({risk_category_h})',
                    'Parkinsons Verdict': parkinson_diagnosis,
                    'Risk of Parkinsons': f'Risk of developing Parkinsons: {risk_score_p:.2f} ({risk_category_p})'
                }

        # Generate Medical Report Button - only enabled after submission
        if 'patient_data' in st.session_state:
            if st.button("Generate Medical Report"):
                patient_data = st.session_state['patient_data']
                print(patient_data)

                pdf = create_pdf(patient_data)
                st.download_button(
                    label="Download Medical Report",
                    data=pdf.getvalue(),  # Get the content of the BytesIO object
                    file_name='medical_report.pdf',
                    mime='application/pdf'
                )


    if selected == 'Health Chatbot':
        st.title("HealthPredictX Chatbot")
        
        st.markdown("""
        Welcome to the HealthPredictX Chatbot. You can ask health-related questions, 
        and the chatbot will provide recommendations based on these conditions. Please note that
        the chatbot may make mistakes.
        """)
        
        # Input for user message
        user_input = st.text_input("You:", placeholder="Type your question here...")

        # Ensure chat history is initialized
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Check if the "Clear Chat" button was clicked
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            user_input = ""  # Clear user input to avoid generating a new response

        # Check if any old entries are tuples and convert them to dictionaries
        for i, chat in enumerate(st.session_state.chat_history):
            if isinstance(chat, tuple):  # If chat is a tuple, convert it to a dictionary
                user_msg, bot_reply = chat
                st.session_state.chat_history[i] = {"user": user_msg, "bot": bot_reply}

        # Only generate a response if user_input is not empty and "Clear Chat" wasn't clicked
        if user_input and 'chat_history' in st.session_state:
            with st.spinner('Generating response...'):
                # Generate chatbot response using GPT-2
                prompt = f"Patient's query: {user_input}\nHealthcare advice:"
                response = suggestion_generator(prompt, max_length=150, num_return_sequences=1, temperature=0.7)
                chatbot_reply = response[0]['generated_text'].strip()

                # Append to chat history as a dictionary
                st.session_state.chat_history.append({"user": user_input, "bot": chatbot_reply})

        # Display chat history
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")  # Now this will always be a dictionary
            st.markdown(f"**HealthPredictX:** {chat['bot']}")

else:
    choice = st.selectbox("Select an option", ["Login", "Register"])
    if choice == "Login":
        login()
    else:
        register()

if st.session_state['logged_in']:
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""
        st.success("You have been logged out.")