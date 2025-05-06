import pandas as pd
import requests
import json
import pyttsx3
import speech_recognition as sr
import pyaudio

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Initialize speech recognizer
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

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to listen to voice input
def listen(prompt):
    speak(prompt)
    with sr.Microphone() as source:
        print(f"Listening for: {prompt}")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio).strip()
        except sr.UnknownValueError:
            speak("Sorry, I didn’t catch that. Please try again.")
            return None
        except sr.RequestError:
            speak("Sorry, there’s an issue with the speech service.")
            return None

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

# Sentiment analysis based on the entire session
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

# Main function with two-way session
def main():
    speak("Welcome to our parent-teacher meeting!")
    print("AI Teacher - Parent-Teacher Meeting")
    speak("Let’s discuss your child. Please say their name, or say 'exit' to end.")
    
    while True:
        student_name = listen("Please say the student’s name.")
        if not student_name:
            continue
        if student_name.lower() == 'exit':
            speak("Thank you for joining me today. Goodbye!")
            print("Goodbye.")
            break
        
        # Start the conversation
        speak(f"Thanks for coming to talk about {student_name}. How are you feeling about their progress?")
        session_messages = []
        
        # First parent input
        parent_message = listen(f"What would you like to say about {student_name}?")
        if not parent_message:
            continue
        session_messages.append(parent_message)
        print(f"Parent: {parent_message}")
        
        # Teacher’s first response
        response = call_deepseek(student_name, parent_message)
        print(f"Teacher: {response}")
        speak(response)
        session_messages.append(response)
        
        # Second parent input
        parent_followup = listen("What do you think about that?")
        if not parent_followup:
            parent_followup = "I’m not sure."
        session_messages.append(parent_followup)
        print(f"Parent: {parent_followup}")
        
        # Teacher’s second response
        response = call_deepseek(student_name, parent_followup, context=parent_message)
        print(f"Teacher: {response}")
        speak(response)
        session_messages.append(response)
        
        # End the session
        speak(f"That’s all for {student_name} today. Would you like to discuss another student?")
        continue_session = listen("Say yes or no.")
        if continue_session and continue_session.lower() == 'no':
            speak("Thank you for the meeting. Goodbye!")
            sentiment = predict_sentiment(session_messages)
            print(f"\nFinal Sentiment Analysis: {sentiment}")
            print("Goodbye.")
            break
        
        sentiment = predict_sentiment(session_messages)
        print(f"\nSession Sentiment Analysis: {sentiment}")
        print("\n---\n")

if __name__ == "__main__":
    main()