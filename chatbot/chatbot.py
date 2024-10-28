import os
import ssl
import random
import sqlite3
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set up SSL and NLTK data path
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Database connection
conn = sqlite3.connect('chatbot.db')
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS user_responses (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_input TEXT,
                     bot_response TEXT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                     )''')

cursor.execute('''CREATE TABLE IF NOT EXISTS intents (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     tag TEXT,
                     pattern TEXT,
                     response TEXT
                     )''')

conn.commit()

# Load intents from the database
def load_intents():
    cursor.execute('SELECT tag, pattern, response FROM intents')
    intents_db = cursor.fetchall()
    intents = {}
    for tag, pattern, response in intents_db:
        if tag not in intents:
            intents[tag] = {"patterns": [], "responses": []}
        intents[tag]["patterns"].append(pattern)
        intents[tag]["responses"].append(response)
    return intents

# Initial intents data (for first-time setup)
initial_intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    }
]

# Function to insert initial intents into the database
def insert_initial_intents():
    for intent in initial_intents:
        tag = intent['tag']
        for pattern in intent['patterns']:
            for response in intent['responses']:
                cursor.execute('INSERT INTO intents (tag, pattern, response) VALUES (?, ?, ?)', (tag, pattern, response))
    conn.commit()

# Uncomment this line to insert initial intents into the database if running for the first time
# insert_initial_intents()

intents = load_intents()

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for tag, data in intents.items():
    for pattern in data['patterns']:
        tags.append(tag)
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Store user responses in the database
def store_user_response(user_input, bot_response):
    cursor.execute('INSERT INTO user_responses (user_input, bot_response) VALUES (?, ?)', (user_input, bot_response))
    conn.commit()

# Chatbot function
def chatbot(input_text):
    input_text_vectorized = vectorizer.transform([input_text])
    tag = clf.predict(input_text_vectorized)[0]
    response = None
    for intent_tag, data in intents.items():
        if intent_tag == tag:
            response = random.choice(data['responses'])
            break
    store_user_response(input_text, response)
    return response

# Streamlit app
def main():
    st.title("Enhanced Chatbot")
    st.write("Welcome to the enhanced chatbot. Please type a message and press Enter to start the conversation.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:")

    if user_input:
        response = chatbot(user_input)
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"Chatbot: {response}")

        # Display chat history
        st.text_area("Chat History", value="\n".join(st.session_state.chat_history), height=150, max_chars=None, key="chat_history")

        # Clear the input box after submission
        st.text_input("You:", value="", key=f"user_input_{len(st.session_state.chat_history)}")


        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
