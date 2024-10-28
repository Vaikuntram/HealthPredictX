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

st.set_page_config(page_title="HealthPredictX",
                   layout="wide",
                   page_icon="🧑‍⚕️")


if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def create_pdf(patient_data):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Comprehensive Medical Report", ln=True, align="C")
    pdf.ln(10) 
  
    pdf.cell(200, 10, txt=f"Name: {patient_data['Name']}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {patient_data['Age']}", ln=True)
    pdf.cell(200, 10, txt=f"Sex: {patient_data['Sex']}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Diabetes Assessment:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"  Verdict: {patient_data['Diabetes Verdict']}", ln=True)
    pdf.cell(200, 10, txt=f"  Risk: {patient_data['Risk of Diabetes']}", ln=True)
    pdf.cell(200, 10, txt=f"  Treatment Suggestion: {patient_data['Diabetes Treatment Suggestion']}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Heart Disease Assessment:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"  Verdict: {patient_data['Heart Disease Verdict']}", ln=True)
    pdf.cell(200, 10, txt=f"  Risk: {patient_data['Risk of Heart Disease']}", ln=True)
    pdf.cell(200, 10, txt=f"  Treatment Suggestion: {patient_data['Heart Disease Treatment Suggestion']}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Parkinson's Disease Assessment:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"  Verdict: {patient_data['Parkinsons Verdict']}", ln=True)
    pdf.cell(200, 10, txt=f"  Risk: {patient_data['Risk of Parkinsons']}", ln=True)
    pdf.cell(200, 10, txt=f"  Treatment Suggestion: {patient_data['Parkinsons Treatment Suggestion']}", ln=True)
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
    user = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Register"):
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT username FROM users WHERE username = ?", (user,))
        if cursor.fetchone():
            st.warning("Username already exists!")
        else:
            register_user(user, password)
            st.success("User registered successfully! You can now log in.")
        connection.close()

def login():
    st.title("Login")
    user = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Login"):
        if login_user(user, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = user
            st.success(f"Welcome, {user}!")
            st.rerun()
        else:
            st.warning("Invalid username or password.")

if st.session_state['logged_in']:
    suggestion_generator = pipeline('text-generation', model='gpt2', device=0, max_length=50)

    def get_ai_health_suggestions(prediction, risk_level, disease):
        
        prompt = (f"The patient's prediction for {disease} is {'positive' if prediction==1 else 'negative'} and their risk level is {risk_level}. "
              "Based on this, provide a recommendation on whether further testing is required.")
    
        generated_text = suggestion_generator(prompt, num_return_sequences=1)[0]['generated_text']
        response_start = prompt.split('recommendation regarding further testing?')[0]
        response = generated_text.replace(response_start, '').strip()
        
        sentences = response.split('. ')
        result = '. '.join(sentences[:2]).strip()
        
        return result if result.endswith('.') else result + '.'
    
    def identify_condition_in_query(query):
        conditions = ["diabetic", "heart disease", "parkinsons"]
        
        for condition in conditions:
            if condition.lower() in query.lower():
                return condition
        return None

    workdir = os.path.dirname(os.path.abspath(__file__))

    diabetesm = pickle.load(open(f'{workdir}/svc_diabetes.sav','rb'))
    heartdiseasem = pickle.load(open(f'{workdir}/logistic_model_updated.sav', 'rb'))
    parkinsonsm = pickle.load(open(f'{workdir}/rf_model_updated.sav', 'rb'))

    with st.sidebar:
        selected = option_menu('HealthPredictX',

                            ['Patient Data','Disease Predictions','Health Chatbot'],
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

            search_name = st.text_input("Enter the ID of the patient")
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
            bp = ''
            skth = ''
            insulin = ''
            bmi = ''
            diabped = ''

            with col1:
                patient_id = st.text_input('Patient ID')

            with col2:
                glucose = st.text_input('Glucose Level')

            with col3:
                bp = st.text_input('Blood Pressure value')

            with col1:
                skth = st.text_input('Skin Thickness value')

            with col2:
                insulin = st.text_input('Insulin Level')

            with col3:
                bmi = st.text_input('BMI value')

            with col1:
                diabped = st.text_input('Diabetes Pedigree Function value')

            if(st.button("Submit")):
                if patient_id and pregnancies and glucose and bp and skth and insulin and bmi and diabped:
                    result = insert_diabetes_data(patient_id, pregnancies, glucose, bp, skth, insulin, bmi, diabped)
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
            freq_start = ''
            freq_high = ''
            freq_low = ''
            jitter_pct = ''
            jitter_abs = ''
            rap_ratio = ''
            ppq_ratio = ''
            ddp_jitter = ''
            shimmer_val = ''
            shimmer_db = ''
            apq_3 = ''
            apq_5 = ''
            apq_val = ''
            dda_shimmer = ''
            noise_ratio = ''
            harmonic_noise_ratio = ''
            rpde_value = ''
            dfa_value = ''
            spread_one = ''
            spread_two = ''
            d2_measure = ''
            ppe_value = ''

            with col1:
                freq_start = st.text_input('MDVP:Fo(Hz)')

            with col2:
                freq_high = st.text_input('MDVP:Fhi(Hz)')

            with col3:
                freq_low = st.text_input('MDVP:Flo(Hz)')

            with col4:
                jitter_pct = st.text_input('MDVP:Jitter(%)')

            with col5:
                jitter_abs = st.text_input('MDVP:Jitter(Abs)')

            with col1:
                rap_ratio = st.text_input('MDVP:RAP')

            with col2:
                ppq_ratio = st.text_input('MDVP:PPQ')

            with col3:
                ddp_jitter = st.text_input('Jitter:DDP')

            with col4:
                shimmer_val = st.text_input('MDVP:Shimmer')

            with col5:
                shimmer_db = st.text_input('MDVP:Shimmer(dB)')

            with col1:
                apq_3 = st.text_input('Shimmer:APQ3')

            with col2:
                apq_5 = st.text_input('Shimmer:APQ5')

            with col3:
                apq_val = st.text_input('MDVP:APQ')

            with col4:
                dda_shimmer = st.text_input('Shimmer:DDA')

            with col5:
                noise_ratio = st.text_input('NHR')

            with col1:
                harmonic_noise_ratio = st.text_input('HNR')

            with col2:
                rpde_value = st.text_input('RPDE')

            with col3:
                dfa_value = st.text_input('DFA')

            with col4:
                spread_one = st.text_input('spread1')

            with col5:
                spread_two = st.text_input('spread2')

            with col1:
                d2_measure = st.text_input('D2')

            with col2:
                ppe_value = st.text_input('PPE')
            
            with col3:
                patient_id = st.text_input('Patient ID')

            if st.button("Submit"):
                if (patient_id and freq_start and freq_high and freq_low and jitter_pct and jitter_abs and rap_ratio 
                    and ppq_ratio and ddp_jitter and shimmer_val and shimmer_db and apq_3 and apq_5 and apq_val 
                    and dda_shimmer and noise_ratio and harmonic_noise_ratio and rpde_value and dfa_value 
                    and spread_one and spread_two and d2_measure and ppe_value):
    
                    result = insert_parkinsons_data(patient_id, freq_start, freq_high, freq_low, jitter_pct, jitter_abs, 
                                                    rap_ratio, ppq_ratio, ddp_jitter, shimmer_val, shimmer_db, 
                                                    apq_3, apq_5, apq_val, dda_shimmer, noise_ratio, 
                                                    harmonic_noise_ratio, rpde_value, dfa_value, spread_one, 
                                                    spread_two, d2_measure, ppe_value)
                    if result:
                        st.success("Parkisons data successfully stored in the database!")
                else:
                    st.error("Please ensure all fields have been filled in.")
    
    if selected == 'Disease Predictions':
        st.title('Disease Predictions')
        patient_id = st.text_input("Enter the Patient ID")
        name = st.text_input("Enter the name of the patient")
        
        if st.button("Submit"):
            if patient_id:
                patient_result = retrieve_patient_data(name)
                age = patient_result[2]
                sex = '1' if patient_result[3] == 'Female' else '0'
                diabetes_result = retrieve_diabetes_data(patient_id)
                heart_result = retrieve_heart_disease_data(patient_id)
                parkinson_result = retrieve_parkinsons_data(patient_id)
                
                diabetes_input = [diabetes_result[2], diabetes_result[3], diabetes_result[4], diabetes_result[5], diabetes_result[6], diabetes_result[7], diabetes_result[8], age]
                diabetes_input = [float(x) for x in diabetes_input]
                diab_prediction = diabetesm.predict([diabetes_input])
                diab_prob = diabetesm.predict_proba([diabetes_input])

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The patient is diabetic'
                else:
                    diab_diagnosis = 'The patient is not diabetic'

                risk_score_d = diab_prob[0][1]
                risk_category_d = 'Low Risk' if risk_score_d < 0.3 else 'Medium Risk' if 0.3 <= risk_score_d < 0.7 else 'High Risk'
                st.success(diab_diagnosis)
                st.write(f'Risk of developing diabetes: {risk_score_d:.2f} ({risk_category_d})')
                ai_suggestions_d = get_ai_health_suggestions(diab_prediction[0], risk_category_d, 'Diabetes')

                heart_input = [age, heart_result[2], heart_result[3], heart_result[4], heart_result[5], heart_result[6], heart_result[7], heart_result[8], heart_result[9], sex, heart_result[10], heart_result[11]]
                heart_input = [float(x) for x in heart_input]
                heart_prediction = heartdiseasem.predict([heart_input])
                heart_prob = heartdiseasem.predict_proba([heart_input])

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The patient is having heart disease and is at risk of heart failure'
                else:
                    heart_diagnosis = 'The patient does not have any heart disease'
                    
                risk_score_h = heart_prob[0][1]
                risk_category_h = 'Low Risk' if risk_score_h < 0.3 else 'Medium Risk' if 0.3 <= risk_score_h < 0.7 else 'High Risk'
                st.success(heart_diagnosis)
                st.write(f'Risk of developing heart disease: {risk_score_h:.2f} ({risk_category_h})')
                ai_suggestions_h = get_ai_health_suggestions(heart_prediction[0], risk_category_h, 'Heart Disease')

                parkinsons_input = [float(x) for x in parkinson_result[2:24]]
                parkinson_prediction = parkinsonsm.predict([parkinsons_input])
                parkinson_prob = parkinsonsm.predict_proba([parkinsons_input])

                if parkinson_prediction[0] == 1:
                    parkinson_diagnosis = "The patient has Parkinson's disease"
                else:
                    parkinson_diagnosis = "The patient doesn't have Parkinson's disease"
                    
                risk_score_p = parkinson_prob[0][1]
                risk_category_p = 'Low Risk' if risk_score_p < 0.3 else 'Medium Risk' if 0.3 <= risk_score_p < 0.7 else 'High Risk'
                st.success(parkinson_diagnosis)
                st.write(f'Risk of developing Parkinsons: {risk_score_p:.2f} ({risk_category_p})')
                ai_suggestions_p = get_ai_health_suggestions(parkinson_prediction[0], risk_category_p, 'Parkinsons')

                st.session_state['patient_data'] = {
                    'Name': patient_result[1],
                    'Age': age,
                    'Sex': patient_result[3],
                    'Diabetes Verdict': diab_diagnosis,
                    'Risk of Diabetes': f'Risk of developing diabetes: {risk_score_d:.2f} ({risk_category_d})',
                    'Diabetes Treatment Suggestion': ai_suggestions_d,
                    'Heart Disease Verdict': heart_diagnosis,
                    'Risk of Heart Disease': f'Risk of developing heart disease: {risk_score_h:.2f} ({risk_category_h})',
                    'Heart Disease Treatment Suggestion': ai_suggestions_h,
                    'Parkinsons Verdict': parkinson_diagnosis,
                    'Risk of Parkinsons': f'Risk of developing Parkinsons: {risk_score_p:.2f} ({risk_category_p})',
                    'Parkinsons Treatment Suggestion': ai_suggestions_p
                }

        if 'patient_data' in st.session_state:
            if st.button("Generate Medical Report"):
                patient_data = st.session_state['patient_data']
                print(patient_data)

                pdf = create_pdf(patient_data)
                st.download_button(
                    label="Download Medical Report",
                    data=pdf.getvalue(), 
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

        tab1, tab2 = st.tabs(["General Queries", "Patient-Specific Queries"])

        with tab1:
            st.header("General Health Queries")

            user_input = st.text_input("You:", placeholder="Type your general health question here...")

            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            if st.button("Clear Chat "):
                st.session_state.chat_history = []
                user_input = "" 

            if user_input and 'chat_history' in st.session_state:
                with st.spinner('Generating response...'):
                    prompt = f"Patient's query: {user_input}\nHealthcare advice:"
                    response = suggestion_generator(prompt, max_length=150, num_return_sequences=1, temperature=0.7)
                    chatbot_reply = response[0]['generated_text'].strip()

                    st.session_state.chat_history.append({"user": user_input, "bot": chatbot_reply})

            for chat in st.session_state.chat_history:
                st.markdown(f"**You:** {chat['user']}")
                st.markdown(f"**HealthPredictX:** {chat['bot']}")

        with tab2:
            st.header("Patient-Specific Queries")

            st.markdown("""
            **How to Ask Patient-Specific Questions:**
            - Please enter the Patient ID for the individual you want to inquire about.
            - Use specific keywords in your questions to get relevant information. For example:
                - "Is the patient diabetic?"
                - "What are the symptoms of heart disease for this patient?"
                - "Does the patient have Parkinsons disease?"
            - Ensure your questions clearly mention the health condition you are inquiring about.
            """)

            patient_id = st.text_input("Enter Patient ID:")
            patient_query = st.text_input("You (Patient Query):", placeholder="Type your question for the patient...")

            if 'patient_chat_history' not in st.session_state:
                st.session_state.patient_chat_history = []

            if st.button("Clear Chat"):
                st.session_state.patient_chat_history = []
                patient_query = "" 

            if patient_id and patient_query and 'patient_chat_history' in st.session_state:
                with st.spinner('Fetching patient data and generating response...'):

                    condition = identify_condition_in_query(patient_query)

                    condition_to_db = {
                        "diabetic": retrieve_diabetes_data,
                        "heart disease": retrieve_heart_disease_data,
                        "parkinsons": retrieve_parkinsons_data
                    }

                    if condition and condition in condition_to_db:

                        patient_data = condition_to_db[condition](patient_id)
                        
                        if patient_data:
                            prompt = f"Patient ID: {patient_id}, Query: {patient_query}\nPatient Data: {patient_data}\nHealthcare advice:"
                            response = suggestion_generator(prompt, max_length=150, num_return_sequences=1)
                            chatbot_reply = response[0]['generated_text'].strip()
                            st.session_state.patient_chat_history.append({"patient_id": patient_id, "user": patient_query, "bot": chatbot_reply})
                        else:
                            chatbot_reply = f"Sorry, no data found for Patient ID {patient_id} related to {condition}."
                    else:
                        chatbot_reply = "Sorry, I'm not sure which condition you're asking about. Please try again."

            for chat in st.session_state.patient_chat_history:
                st.markdown(f"**Patient ID {chat['patient_id']} - You:** {chat['user']}")
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
