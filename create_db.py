import sqlite3
from datetime import datetime

conn = sqlite3.connect('health_db.sqlite')
c = conn.cursor()
c.execute("SELECT * FROM patients")
patients = c.fetchall()
for patient in patients:
    print(patient)

with sqlite3.connect('health_db.sqlite') as connection:  # replace with your database path
        cursor = connection.cursor()
        
        # Run the PRAGMA command
        cursor.execute("PRAGMA table_info(parkinsons_data_1);")
        
        # Fetch all results
        schema_info = cursor.fetchall()
        
        # Print the schema information
        for column in schema_info:
            print(column)

def create_connection():
    connection = sqlite3.connect('health_db.sqlite', check_same_thread=False, timeout=5)
    return connection

def create_tables():
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        age INTEGER,
                        gender TEXT,
                        address TEXT,
                        phone_number TEXT,
                        email TEXT,
                        created_at TEXT
                      );''')

    
    c.execute('''CREATE TABLE IF NOT EXISTS diabetes_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id INTEGER,
                        pregnancies INTEGER,
                        glucose INTEGER,
                        blood_pressure INTEGER,
                        skin_thickness INTEGER,
                        insulin INTEGER,
                        bmi REAL,
                        diabetes_pedigree REAL,
                        created_at TEXT,
                        FOREIGN KEY(patient_id) REFERENCES patients(id)
                      );''')

    c.execute('''CREATE TABLE IF NOT EXISTS heart_disease_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id INTEGER,
                        anaemia INTEGER CHECK (anaemia IN (0,1)),
                        creatine INTEGER,
                        diabetes INTEGER CHECK (diabetes IN (0,1)),
                        ejection_fraction INTEGER,
                        bp INTEGER CHECK (bp IN (0,1)),
                        platelets INTEGER,
                        serum_creatinine REAL,
                        serum_sodium INTEGER,
                        smoking INTEGER CHECK (smoking IN (0,1)),
                        follow_up INTEGER,
                        FOREIGN KEY(patient_id) REFERENCES patients(id)
                      );''')

    c.execute('''CREATE TABLE IF NOT EXISTS parkinsons_data_1 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id INTEGER,
                        fo REAL,  -- MDVP:Fo(Hz)
                        fhi REAL, -- MDVP:Fhi(Hz)
                        flo REAL, -- MDVP:Flo(Hz)
                        jitter_percent REAL, -- MDVP:Jitter(%)
                        jitter_abs REAL, -- MDVP:Jitter(Abs)
                        rap REAL, -- MDVP:RAP
                        ppq REAL, -- MDVP:PPQ
                        ddp REAL, -- Jitter:DDP
                        shimmer REAL, -- MDVP:Shimmer
                        shimmer_db REAL, -- MDVP:Shimmer(dB)
                        apq3 REAL, -- Shimmer:APQ3
                        apq5 REAL, -- Shimmer:APQ5
                        apq REAL, -- MDVP:APQ
                        dda REAL, -- Shimmer:DDA
                        nhr REAL, -- NHR
                        hnr REAL, -- HNR
                        rpde REAL, -- RPDE
                        dfa REAL, -- DFA
                        spread1 REAL, -- spread1
                        spread2 REAL, -- spread2
                        d2 REAL, -- D2
                        ppe REAL, -- PPE
                        created_at TEXT,
                        FOREIGN KEY(patient_id) REFERENCES patients(id)
                      );''')

    conn.commit()

def insert_patient_data(name, age, gender, address, phone_number, email):
    with create_connection() as connection:
        try:
            c = connection.cursor()
            query = "INSERT INTO patients (name, age, gender, address, phone_number, email, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
            c.execute(query, (name, int(age), gender, address, phone_number, email, datetime.now()))
            connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return False
        
def retrieve_patient_data(name):
    with create_connection() as connection:
        try:
            c = connection.cursor()
            query = "SELECT * FROM patients WHERE id = ?"
            c.execute(query, (name,))
            patient_data = c.fetchone()
            return patient_data
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return None

def insert_diabetes_data(patient_id, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree):
    with create_connection() as connection:
        try:
            c = connection.cursor()
            query = '''INSERT INTO diabetes_data (patient_id, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, created_at) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''
            c.execute(query, (patient_id, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, datetime.now()))
            connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return False

def retrieve_diabetes_data(patient_id):
    with create_connection() as connection:
        try:
            c = connection.cursor()
            query = "SELECT * FROM diabetes_data WHERE patient_id = ?"
            c.execute(query, (patient_id,))
            diabetes_data = c.fetchone()
            return diabetes_data
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return None

def insert_heart_disease_data(patient_id, anaemia, creatine, diabetes, ejection_fraction, bp, platelets, serum_creatinine, serum_sodium, smoking, follow_up):
    with create_connection() as connection:
        try:
            c = connection.cursor()
            query = '''INSERT INTO heart_disease_data (patient_id, anaemia, creatine, diabetes, ejection_fraction, bp, platelets, serum_creatinine, serum_sodium, smoking, follow_up) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
            c.execute(query, (patient_id, anaemia, creatine, diabetes, ejection_fraction, bp, platelets, serum_creatinine, serum_sodium, smoking, follow_up))
            connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return False

def retrieve_heart_disease_data(patient_id):
    with create_connection() as connection:
        try:
            c = connection.cursor()
            query = "SELECT * FROM heart_disease_data WHERE patient_id = ?"
            c.execute(query, (patient_id,))
            heart_disease_data = c.fetchone()
            return heart_disease_data
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return None

def insert_parkinsons_data(patient_id, fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe):
    with create_connection() as connection:
        try:
            c = connection.cursor()


            query = '''INSERT INTO parkinsons_data_1 (
             patient_id, fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, 
             shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, 
             spread1, spread2, d2, ppe, created_at) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

            c.execute(query, (patient_id, fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                              shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, 
                              spread1, spread2, d2, ppe, datetime.now()))  # Ensure this has 25 items.


            connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return False

def retrieve_parkinsons_data(patient_id):
    with create_connection() as connection:
        try:
            c = connection.cursor()
            query = "SELECT * FROM parkinsons_data_1 WHERE patient_id = ?"
            c.execute(query, (patient_id,))
            parkinsons_data = c.fetchone()
            return parkinsons_data
        except sqlite3.Error as e:
            print(f"Error: {e}")
            return None
