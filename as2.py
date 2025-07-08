import tkinter as tk
import sqlite3
import cv2
from fer import FER
import pyttsx3
import time
import speech_recognition as sr
from pynput import keyboard
import threading
import queue
import datetime

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Lock to control access to pyttsx3 engine
tts_lock = threading.Lock()

# Initialize facial emotion recognition model
emotion_detector = FER()

# Initialize speech recognition
recognizer = sr.Recognizer()

# Global flag to stop the program
stop_program = False

# Queue for database operations
db_queue = queue.Queue()

# Initialize the database and create tables if they don't exist
def initialize_database():
    conn = sqlite3.connect('patient_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
                      name TEXT PRIMARY KEY,
                      age INTEGER,
                      disease TEXT,
                      allergic_food TEXT,
                      schedule TEXT,
                      food_intake INTEGER,
                      water_intake INTEGER)''')
    conn.commit()
    conn.close()

# Function to handle database operations
def database_worker():
    conn = sqlite3.connect('patient_data.db')
    cursor = conn.cursor()
    while not stop_program:
        try:
            operation = db_queue.get(timeout=1)  # Get database operation from the queue
            if operation:
                op_type, *args = operation
                if op_type == 'insert':
                    cursor.execute("INSERT INTO patients (name, age, disease, allergic_food, schedule, food_intake, water_intake) VALUES (?, ?, ?, ?, ?, ?, ?)", args)
                elif op_type == 'update_food':
                    cursor.execute("UPDATE patients SET food_intake = ? WHERE name = ?", args)
                elif op_type == 'update_water':
                    cursor.execute("UPDATE patients SET water_intake = ? WHERE name = ?", args)
                elif op_type == 'select':
                    cursor.execute("SELECT * FROM patients WHERE name = ?", args)
                    result = cursor.fetchone()
                    db_queue.put(result)  # Put the result back in the queue for processing
                conn.commit()
        except queue.Empty:
            continue
    conn.close()

# Function to collect and store patient data
def submit_form():
    global food_counter, water_counter
    name = name_entry.get()
    age = age_entry.get()
    disease = disease_entry.get()
    allergic_food = allergic_food_entry.get()
    patient_schedule = schedule_entry.get()

    # Reset counters on new entry
    food_counter = 0
    water_counter = 0

    # Add database operation to the queue
    db_queue.put(('insert', name, age, disease, allergic_food, patient_schedule, food_counter, water_counter))

    # Confirmation message
    print(f"Saved data for {name}")

    # Start monitoring tasks
    start_monitoring()

# Continuous listening for speech and process food and water intake
def listen_for_food_intake():
    global food_counter, water_counter
    while not stop_program:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)  # Adjust microphone to ambient noise
            print("Listening for food, water intake or details/status...")

            try:
                audio = recognizer.listen(source, timeout=5)  # Adjust timeout as needed
                intake_statement = recognizer.recognize_google(audio)
                print(f"Patient said: {intake_statement}")
                process_intake(intake_statement)
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
                with tts_lock:
                    engine.say("Sorry, I did not understand that.")
                    engine.runAndWait()
            except sr.RequestError as e:
                print(f"Error with speech recognition service; {e}")
                with tts_lock:
                    engine.say("There was an issue with the speech recognition service.")
                    engine.runAndWait()

# Function to process intake and update the database
def process_intake(intake_statement):
    global food_counter, water_counter
    name = name_entry.get()  # Get the current patient's name

    # Refined keyword lists for better detection
    food_keywords = ['ate', 'food', 'meal', 'snack', 'breakfast', 'lunch', 'dinner']
    water_keywords = ['water', 'drank', 'drink', 'hydration']
    allergic_foods = allergic_food_entry.get().split(',')

    # Commands for showing details or status
    if "show my details" in intake_statement.lower():
        db_queue.put(('select', name))
        result = db_queue.get()  # Get the result from the queue
        if result:
            try:
                details = f"Name: {result[0]}, Age: {result[1]}, Disease: {result[2]}, Allergic Foods: {result[3]}, Schedule: {result[4]}"
                with tts_lock:
                    engine.say(details)
                    engine.runAndWait()
                print(details)
            except IndexError:
                with tts_lock:
                    engine.say("Error retrieving details from the database.")
                    engine.runAndWait()
                print("Error retrieving details from the database.")
    elif "show my status" in intake_statement.lower():
        status = f"Food intake: {food_counter} times, Water intake: {water_counter} times"
        with tts_lock:
            engine.say(status)
            engine.runAndWait()
        print(status)
    elif any(word in intake_statement.lower() for word in food_keywords):
        food_counter += 1  # Increment the food intake counter
        db_queue.put(('update_food', food_counter, name))
        with tts_lock:
            engine.say(f"Food intake recorded. This is meal number {food_counter}.")
            engine.runAndWait()
        print(f"Updated food intake: {intake_statement}")

        # Check for allergic foods
        if any(allergic_food.lower() in intake_statement.lower() for allergic_food in allergic_foods):
            with tts_lock:
                engine.say("Warning: The intake includes an allergic food.")
                engine.runAndWait()
            print("Alert: Allergic food detected!")
    elif any(word in intake_statement.lower() for word in water_keywords):
        water_counter += 1  # Increment the water intake counter
        db_queue.put(('update_water', water_counter, name))
        with tts_lock:
            engine.say(f"Water intake recorded. This is drink number {water_counter}.")
            engine.runAndWait()
        print(f"Updated water intake: {intake_statement}")
    else:
        with tts_lock:
            engine.say("I did not detect food or water intake in your statement.")
            engine.runAndWait()

# Emotion detection every 10 minutes
def detect_emotion_and_interact():
    cap = cv2.VideoCapture(0)
    while not stop_program:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break

        emotion_scores = emotion_detector.detect_emotions(frame)
        if emotion_scores:
            # Extract emotions from the score
            emotions = emotion_scores[0]['emotions']
            if emotions:
                top_emotion = max(emotions, key=emotions.get)  # Find emotion with the highest score
                score = emotions[top_emotion]
                print(f"Detected emotion: {top_emotion} with score {score}")
                cv2.putText(frame, f"Emotion: {top_emotion} (Score: {score})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Emotion Detection', frame)

                message = ""
                if top_emotion == 'happy':
                    message = "You seem happy! Keep smiling!"
                elif top_emotion == 'sad':
                    message = "I see you're feeling sad."
                elif top_emotion == 'angry':
                    message = "I sense some frustration. Try to relax."
                elif top_emotion == 'disgust':
                    message = "It looks like something is bothering you."
                elif top_emotion == 'neutral':
                    message = "I see you are feeling neutral."

                if message:
                    with tts_lock:
                        engine.say(f"{name_entry.get()}, {message}")
                        engine.runAndWait()
        
        if cv2.waitKey(1) & 0xFF == ord('q') or stop_program:
            break

        # Wait for 10 minutes before detecting mood again
        print("Waiting for 10 minutes before the next emotion detection...")
        time.sleep(600)  # Sleep for 600 seconds (10 minutes)

    cap.release()
    cv2.destroyAllWindows()

# Alarm system to notify the patient at the specified time
def check_alarm():
    while not stop_program:
        current_time = datetime.datetime.now().strftime("%H:%M")
        name = name_entry.get()

        # Add database operation to the queue to fetch the schedule
        db_queue.put(('select', name))
        result = db_queue.get()  # Get the result from the queue
        if result:
            try:
                schedule = result[4]  # Retrieve schedule from the result
                if schedule:
                    # Example schedule format: "09:00, 12:00, 18:00"
                    alarms = schedule.split(', ')
                    if current_time in alarms:
                        with tts_lock:
                            engine.say(f"It's time for your scheduled activity.")
                            engine.runAndWait()
                        print(f"Alarm triggered for time: {current_time}")
            except IndexError:
                print("Error checking schedule: tuple index out of range")
        
        time.sleep(60)  # Check every minute

# Start the program
def start_monitoring():
    # Initialize the database
    initialize_database()

    # Start the database worker thread
    db_thread = threading.Thread(target=database_worker, daemon=True)
    db_thread.start()

    # Start emotion detection thread
    emotion_thread = threading.Thread(target=detect_emotion_and_interact, daemon=True)
    emotion_thread.start()

    # Start alarm check thread
    alarm_thread = threading.Thread(target=check_alarm, daemon=True)
    alarm_thread.start()

    # Start listening for food and water intake
    listen_thread = threading.Thread(target=listen_for_food_intake, daemon=True)
    listen_thread.start()

    # Run the GUI loop
    root.mainloop()

# GUI setup
root = tk.Tk()
root.title("Patient Monitoring System")

# GUI elements
tk.Label(root, text="Name:").grid(row=0, column=0)
tk.Label(root, text="Age:").grid(row=1, column=0)
tk.Label(root, text="Disease:").grid(row=2, column=0)
tk.Label(root, text="Allergic Foods:").grid(row=3, column=0)
tk.Label(root, text="Schedule (comma-separated):").grid(row=4, column=0)

name_entry = tk.Entry(root)
age_entry = tk.Entry(root)
disease_entry = tk.Entry(root)
allergic_food_entry = tk.Entry(root)
schedule_entry = tk.Entry(root)

name_entry.grid(row=0, column=1)
age_entry.grid(row=1, column=1)
disease_entry.grid(row=2, column=1)
allergic_food_entry.grid(row=3, column=1)
schedule_entry.grid(row=4, column=1)

submit_button = tk.Button(root, text="Submit", command=submit_form)
submit_button.grid(row=5, column=0, columnspan=2)

# Start the main loop
if __name__ == "__main__":
    start_monitoring()
