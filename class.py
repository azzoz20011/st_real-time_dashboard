import sqlite3
import time
import random
import streamlit as st
import threading

# Database setup
def create_db():
    conn = sqlite3.connect("data.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")  # Enable WAL mode for concurrency
    c.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            current_speed REAL,
            distance REAL,
            status BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_data(speed, distance, status):
    conn = sqlite3.connect("data.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO telemetry (current_speed, distance, status) VALUES (?, ?, ?)", (speed, distance, status))
    conn.commit()
    conn.close()

def get_latest_data():
    conn = sqlite3.connect("data.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT current_speed, distance, status FROM telemetry ORDER BY id DESC LIMIT 1")
    data = c.fetchone()
    conn.close()
    return data if data else (0, 0, False)

# Simulated data generation
def generate_data():
    while True:
        speed = round(random.uniform(0, 100), 2)
        distance = round(random.uniform(0, 500), 2)
        status = random.choice([True, False])
        insert_data(speed, distance, status)
        time.sleep(0.1)

# Streamlit Dashboard
def dashboard():
    st.title("Real-Time Telemetry Dashboard")
    
    speed_placeholder = st.empty()
    distance_placeholder = st.empty()
    status_placeholder = st.empty()
    
    while True:
        speed, distance, status = get_latest_data()
        speed_placeholder.metric(label="Current Speed (km/h)", value=speed)
        distance_placeholder.metric(label="Distance (m)", value=distance)
        status_placeholder.metric(label="Status", value="Active" if status else "Inactive")
        
        time.sleep(0.5)  # Allow refresh
        st.rerun()  # Updated function for Streamlit rerun

if __name__ == "__main__":
    create_db()
    threading.Thread(target=generate_data, daemon=True).start()
    dashboard()
