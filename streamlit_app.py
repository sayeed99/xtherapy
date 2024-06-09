import streamlit as st
import time
from client import complete
import uuid
import json
import os
import threading

st.title("AI Therapy Assistant : Anna")

# Global lock for synchronizing access to the append_to_json function
json_lock = threading.Lock()

# Initialize chat history
if "messages" not in st.session_state:
    print("setting messages")
    st.session_state.messages = []

if "code" not in st.session_state:
    print("setting code")
    st.session_state.code = str(uuid.uuid4()) + ".json"

def load_json(filename):
    """Loads and returns the content of a JSON file."""
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"The file {filename} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {filename}.")
        return None

def load_conversation():
    file = "conversations/f03f8c17-8388-4b3e-bde8-3d316a429a22.json"
    data = load_json(file)
    for datum in data:
        st.session_state.messages.append(datum)

# load_conversation()

def stream_data(response):
    for word in response:
        yield word + ""
        time.sleep(0.02)

def save_messages():
    folder_path = "conversations"
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, st.session_state.code)
    
    with json_lock:
        with open(file_path, 'w') as file:
            json.dump(st.session_state.messages, file, indent=4)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hi"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_messages()  # Save messages after user input
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        ai_response = complete(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        save_messages()  # Save messages after assistant response
        # response = st.write(ai_response)

        response = st.write_stream(stream_data(ai_response))
