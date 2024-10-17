import sqlite3

def create_connection():
    connection = sqlite3.connect('user_db.sqlite')
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                      );''')
    connection.commit()
    return connection

def register_user(username, password):
    connection = create_connection()
    try:
        cursor = connection.cursor()
        query = "INSERT INTO users (username, password) VALUES (?, ?)"
        cursor.execute(query, (username, password))
        connection.commit()
        print("User registered successfully!")
    except sqlite3.Error as e:
        print(f"Error registering user: {e}")
    finally:
        connection.close()

def login_user(username, password):
    connection = create_connection()
    try:
        cursor = connection.cursor()
        query = "SELECT * FROM users WHERE username = ? AND password = ?"
        cursor.execute(query, (username, password))
        result = cursor.fetchone()
        return bool(result)
    except sqlite3.Error as e:
        print(f"Error logging in: {e}")
    finally:
        connection.close()
