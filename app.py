import pandas as pd
import requests
import json
import pyttsx3
import speech_recognition as sr
import flask
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

# Initialize text-to-speech engine (for server-side fallback, optional)
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize speech recognizer (optional, not used in web context)
recognizer = sr.Recognizer()

# Load the CSV file
df = pd.read_csv('student_profiles.csv')

# OpenRouter API configuration
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Session state to track conversations
sessions = {}  # Dictionary to store session data: {session_id: {student_name, messages, context}}

# Function to call DeepSeek via OpenRouter
def call_deepseek(student_name, parent_message, context=""):
    student = df[df['Name'].str.lower() == student_name.lower()]
    if student.empty:
        return "I couldn’t find that student. Check the name and try again."
    
    strength = student['Strength'].values[0]
    weakness = student['Weakness'].values[0]
    trait = student['Personality_Trait'].values[0]
    
    prompt = (
        f"You are an AI teacher in a parent-teacher meeting. Student: {student_name}, "
        f"Strength: {strength}, Weakness: {weakness}, Trait: {trait}. "
        f"Parent says: '{parent_message}'. Context: '{context}'. "
        f"Give a short, realistic response acknowledging the strength, addressing the message, "
        f"and asking a follow-up question to keep the conversation going."
    )
    
    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful AI teacher in a meeting."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        return f"Error with AI: {str(e)}"

# Sentiment analysis
def predict_sentiment(session_messages):
    message_lower = " ".join(session_messages).lower()
    positive_words = ['great', 'good', 'happy', 'proud', 'excellent', 'improving']
    negative_words = ['worried', 'bad', 'struggle', 'poor', 'concern', 'issue']
    pos_count = sum(word in message_lower for word in positive_words)
    neg_count = sum(word in message_lower for word in negative_words)
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral"

# Flask Routes
@app.route('/start', methods=['POST'])
def start_meeting():
    session_id = str(len(sessions) + 1)  # Simple session ID generation
    sessions[session_id] = {"student_name": None, "messages": [], "context": ""}
    return jsonify({"message": "Welcome to our parent-teacher meeting! Please provide the student’s name."})

@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    message = data.get('message', '').strip()
    session_id = data.get('session_id', '1')  # Default to '1' for simplicity; improve with proper session management
    
    if session_id not in sessions:
        return jsonify({"response": "Session not found. Please start a new meeting."}), 400
    
    session = sessions[session_id]
    
    # If no student name is set, assume the first message is the student name
    if not session["student_name"]:
        student_name = message
        student = df[df['Name'].str.lower() == student_name.lower()]
        if student.empty:
            return jsonify({"response": "I couldn’t find that student. Check the name and try again."})
        session["student_name"] = student_name
        session["messages"].append(message)
        response = f"Thanks for coming to talk about {student_name}. How are you feeling about their progress?"
        session["messages"].append(response)
        return jsonify({"response": response})
    
    # Process parent message
    session["messages"].append(message)
    response = call_deepseek(session["student_name"], message, context=session["context"])
    session["messages"].append(response)
    session["context"] = message  # Update context with the latest parent message
    return jsonify({"response": response})

@app.route('/end', methods=['POST'])
def end_meeting():
    data = request.get_json()
    session_id = data.get('session_id', '1')
    
    if session_id not in sessions:
        return jsonify({"sentiment": "Neutral", "message": "Session not found."}), 400
    
    session = sessions[session_id]
    sentiment = predict_sentiment(session["messages"])
    del sessions[session_id]  # Clear session
    return jsonify({"sentiment": sentiment, "message": "Thank you for the meeting. Goodbye!"})
@app.route('/')
def serve_index():
    return app.send_static_file('index.html')
# Optional: Run the original console version
def run_console():
    main()  # Original main() function from the provided script

if __name__ == "__main__":
    # Start Flask server
    app.run(debug=True, port=5000)
    # Optionally, run console version in a separate thread
    # threading.Thread(target=run_console).start()