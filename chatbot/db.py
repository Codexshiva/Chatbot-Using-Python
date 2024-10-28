import sqlite3

# Connect to the database
conn = sqlite3.connect('chatbot.db')
cursor = conn.cursor()

# Example: Fetch all user responses
cursor.execute('SELECT * FROM user_responses')
user_responses = cursor.fetchall()
for response in user_responses:
    print(response)

# Example: Fetch all intents
cursor.execute('SELECT * FROM intents')
intents = cursor.fetchall()
for intent in intents:
    print(intent)

# Closing the connection
conn.close()
