import os
import torch
import requests
import logging
import asyncio
import streamlit as st
import numpy as np
import boto3
import threading
import pytesseract
from fastapi import FastAPI, WebSocket, UploadFile
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, ViTForImageClassification
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from groq import Groq
from time import sleep
from streamlit_js_eval import streamlit_js_eval
import uvicorn
from threading import Thread

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# Validate API Keys
if not GROQ_API_KEY or not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("❌ ERROR: Missing API keys! Please check your .env file or Streamlit secrets.")

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)
polly_client = boto3.client("polly", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# ------------------ ✅ FastAPI Backend ------------------
app = FastAPI()

# Chat API Endpoint
@app.post("/chat/")
async def chat(query: dict):
    user_input = query.get("message", "")

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": "You are an AI assistant."},
                  {"role": "user", "content": user_input}],
        model="llama-3.3-70b-versatile"
    )

    response = chat_completion.choices[0].message.content
    return {"response": response}

# Image Processing Endpoint
@app.post("/analyze_image/")
async def analyze_image(file: UploadFile):
    image = Image.open(BytesIO(await file.read()))
    extracted_text = pytesseract.image_to_string(image)
    return {"extracted_text": extracted_text}

# Text-to-Speech (TTS) WebSocket
@app.websocket("/tts/")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text_data = await websocket.receive_text()
            response = polly_client.synthesize_speech(Text=text_data, OutputFormat="mp3", VoiceId="Joanna")
            audio_stream = response["AudioStream"].read()
            await websocket.send_bytes(audio_stream)
    except:
        logging.warning("TTS WebSocket disconnected.")
    finally:
        await websocket.close()

# Start FastAPI inside Streamlit
def start_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

Thread(target=start_fastapi, daemon=True).start()

# ------------------ ✅ Streamlit UI ------------------
st.set_page_config(page_title="ANU.AI", page_icon="🤖", layout="wide")

# Custom CSS Styling for Dark Mode UI
st.markdown("""
    <style>
        body { background-color: #1a1b26; color: white; }
        .main { background-color: #1a1b26; }
        .chat-container { padding: 20px; }
        .chat-bubble {
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 60%;
            display: inline-block;
            margin-bottom: 10px;
        }
        .chat-user { background-color: #2b6cb0; color: white; text-align: right; margin-left: auto; }
        .chat-bot { background-color: #1e293b; color: white; text-align: left; }
        .chat-input { width: 90%; padding: 10px; border-radius: 10px; border: none; background: #2d2f3b; color: white; }
        .sidebar { background-color: #1a1b26; padding: 15px; }
        .sidebar button { width: 100%; margin-bottom: 10px; padding: 10px; border-radius: 10px; background: #2b6cb0; color: white; border: none; }
        .action-btn { background: #2b6cb0; color: white; border: none; padding: 10px; margin-right: 5px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# Chat Header
st.markdown("<h1 style='text-align: center;'>🤖 ANU.AI - Your Smart AI Assistant</h1>", unsafe_allow_html=True)

# Sidebar for Quick Actions
st.sidebar.markdown("## Settings")
st.sidebar.button("📥 Download Chat")
st.sidebar.button("🔗 Share Chat")
st.sidebar.markdown("## Quick Actions")
st.sidebar.button("💻 Help me write code")
st.sidebar.button("📖 Explain a concept")
st.sidebar.button("🎨 Generate ideas")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "bot", "content": "Hello! How can I assist you today? 😊"}]

# Display Chat Messages
for message in st.session_state.chat_history:
    role_class = "chat-user" if message["role"] == "user" else "chat-bot"
    st.markdown(f"<div class='chat-bubble {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

# 🎙️ Voice Input Using JavaScript (Web Speech API)
st.write("🎙️ Click below to use voice input:")

speech_text = streamlit_js_eval(js_expressions="window.navigator.mediaDevices.getUserMedia({ audio: true });", key="speech_recognition")

if speech_text:
    st.session_state.chat_history.append({"role": "user", "content": speech_text})

    # Send to chatbot backend
    try:
        response = requests.post("http://127.0.0.1:8000/chat/", json={"message": speech_text}, timeout=10)
        response.raise_for_status()
        bot_response = response.json().get("response", "I didn't understand that.")
    except requests.exceptions.RequestException as e:
        bot_response = f"⚠️ Error: {str(e)}"

    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    st.markdown(f"**🤖 ANU.AI:** {bot_response}")

# Chat Input
user_input = st.text_input("💬 Type your message here...", key="chat_input")

# Send Button
if st.button("📤 Send", key="send_button"):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        try:
            response = requests.post("http://127.0.0.1:8000/chat/", json={"message": user_input}, timeout=10)
            response.raise_for_status()
            bot_response = response.json().get("response", "I didn't understand that.")
        except requests.exceptions.RequestException as e:
            bot_response = f"⚠️ Error: {str(e)}"

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        st.markdown(f"**🤖 ANU.AI:** {bot_response}")
