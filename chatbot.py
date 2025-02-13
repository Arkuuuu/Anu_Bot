import os
import torch
import requests
import logging
import streamlit as st
import numpy as np
import boto3
import threading
import pytesseract
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from groq import Groq
from uvicorn import run
from streamlit_js_eval import streamlit_js_eval

# ---------------------- Load Environment Variables ----------------------
load_dotenv()

# ---------------------- Setup Logging ----------------------
logging.basicConfig(level=logging.INFO)

# ---------------------- API Keys ----------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PINECONE_INDEX_NAME = "ai-multimodal-chatbot"

# Validate API Keys
if not GROQ_API_KEY or not AWS_ACCESS_KEY or not AWS_SECRET_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ ERROR: Missing API keys! Please check your .env file or Streamlit secrets.")

# ---------------------- Initialize Clients ----------------------
groq_client = Groq(api_key=GROQ_API_KEY)
polly_client = boto3.client("polly", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

# ---------------------- Initialize Pinecone ----------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
available_indexes = [index.name for index in pc.list_indexes()]

if PINECONE_INDEX_NAME in available_indexes:
    logging.info(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    index = pc.Index(PINECONE_INDEX_NAME)
else:
    raise ValueError(f"❌ ERROR: Pinecone index '{PINECONE_INDEX_NAME}' not found. Check your Pinecone dashboard.")

# ---------------------- Load Hugging Face Models ----------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# ---------------------- FastAPI Backend ----------------------
app = FastAPI()

# Enable CORS for frontend-backend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat/")
async def chat(query: dict):
    user_input = query.get("message", "")

    # Get AI response
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": "You are an AI assistant."}, {"role": "user", "content": user_input}],
        model="llama-3.3-70b-versatile"
    )
    
    response = chat_completion.choices[0].message.content
    return {"response": response}

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="ANU.AI", page_icon="🤖", layout="wide")

# ---------------------- Inject Custom CSS ----------------------
st.markdown("""
    <style>
        /* Dark Mode Theme */
        body { background-color: #0d1117; color: white; }
        .main { background-color: #0d1117; }
        
        /* Chat Input Bar at Bottom */
        .chat-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: #161b22;
            padding: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .chat-input {
            flex: 1;
            padding: 10px;
            border-radius: 10px;
            background: #21262d;
            color: white;
            border: none;
            outline: none;
            font-size: 16px;
        }
        .chat-btn {
            width: 40px;
            height: 40px;
            margin-left: 8px;
            border-radius: 50%;
            background: #30363d;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }
        .chat-btn:hover {
            background: #484f58;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Sidebar for Settings ----------------------
with st.sidebar:
    st.title("Settings")
    st.button("📥 Download Chat")
    st.button("🔗 Share Chat")
    st.subheader("Quick Actions")
    st.button("💻 Help me write code")
    st.button("📖 Explain a concept")
    st.button("🎨 Generate ideas")

# ---------------------- Chat Messages ----------------------
st.title("💬 Anu.AI Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------- Chat Input and Buttons ----------------------
col1, col2, col3, col4, col5 = st.columns([1, 8, 1, 1, 1])

# 🎤 Mic Button
with col1:
    mic_clicked = st.button("🎙️")

# 📩 Input Box
with col2:
    user_input = st.text_input("Type your message...", key="chat_input", label_visibility="collapsed")

# 😃 Emoji Button
with col3:
    emoji_clicked = st.button("😊")

# 📎 Upload Button
with col4:
    upload_clicked = st.button("📎")

# 🚀 Send Button
with col5:
    send_clicked = st.button("📤")

# ---------------------- Voice Input Handling ----------------------
if mic_clicked:
    speech_text = streamlit_js_eval(js_expressions="window.navigator.mediaDevices.getUserMedia({ audio: true });", key="speech_recognition")
    
    if speech_text:
        st.session_state.chat_history.append({"role": "user", "content": speech_text})
        st.markdown("🤖 **ANU.AI is analyzing... ⏳**")  
        response = requests.post("http://127.0.0.1:8000/chat/", json={"message": speech_text}, timeout=10)
        bot_response = response.json().get("response", "I didn't understand that.")

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.write(bot_response)

# ---------------------- Send Message Handling ----------------------
if send_clicked and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.markdown("🤖 **ANU.AI is analyzing... ⏳**")  
    response = requests.post("http://127.0.0.1:8000/chat/", json={"message": user_input}, timeout=10)
    bot_response = response.json().get("response", "I didn't understand that.")

    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.write(bot_response)

# ---------------------- Start FastAPI Server ----------------------
def run_fastapi():
    run(app, host="0.0.0.0", port=8000, log_level="info")

threading.Thread(target=run_fastapi, daemon=True).start()
