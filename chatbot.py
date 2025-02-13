import os
import torch
import requests
import logging
import asyncio
import streamlit as st
import numpy as np
import boto3
import websocket
import threading
import pytesseract
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, ViTForImageClassification
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from groq import Groq
from time import sleep
from streamlit_js_eval import streamlit_js_eval

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WHISPER_API_URL = "https://api.groq.com/audio/transcriptions"
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PINECONE_INDEX_NAME = "ai-multimodal-chatbot"

# Validate API Keys
if not GROQ_API_KEY or not AWS_ACCESS_KEY or not AWS_SECRET_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ ERROR: Missing API keys! Check your .env file or Streamlit secrets.")

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)
polly_client = boto3.client("polly", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

# Initialize FastAPI
app = FastAPI()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [index_info['name'] for index_info in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    logging.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    logging.info(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' created successfully!")
else:
    logging.info(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

index = pc.Index(PINECONE_INDEX_NAME)

# Load Hugging Face Models
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Updated ViT Model for Image Processing
vit_model_name = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(vit_model_name)
vit_model = ViTForImageClassification.from_pretrained(vit_model_name)

# -----------------------------------------------------------------FastAPI Backend---------------------------------------------------------------

# Chat Function using Groq API
@app.post("/chat/")
async def chat(query: dict):
    user_input = query.get("message", "")

    # Call Groq API for response
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {"messages": [{"role": "user", "content": user_input}], "model": "llama-3.3-70b-versatile"}

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        bot_response = response.json()["choices"][0]["message"]["content"]
        return {"response": bot_response}
    else:
        return {"response": "⚠️ Error: Unable to connect to Groq API."}

# Speech-to-Text (STT) using Whisper API
@app.websocket("/stt/")
async def websocket_stt(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            audio_buffer.extend(audio_data)

            if len(audio_buffer) > 16000:  
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
                files = {"file": ("audio.wav", audio_buffer, "audio/wav")}
                response = requests.post(WHISPER_API_URL, headers=headers, files=files)

                if response.status_code == 200:
                    transcription = response.json().get("text", "")
                    await websocket.send_text(transcription)
                else:
                    await websocket.send_text("⚠️ Error: STT failed.")
                audio_buffer.clear()
    except WebSocketDisconnect:
        logging.warning("STT WebSocket disconnected.")
    finally:
        await websocket.close()

# Text-to-Speech (TTS) using AWS Polly
@app.websocket("/tts/")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text_data = await websocket.receive_text()
            response = polly_client.synthesize_speech(Text=text_data, OutputFormat="mp3", VoiceId="Joanna")
            audio_stream = response["AudioStream"].read()
            await websocket.send_bytes(audio_stream)
    except WebSocketDisconnect:
        logging.warning("TTS WebSocket disconnected.")
    finally:
        await websocket.close()

# -----------------------------------------------------------------Streamlit UI---------------------------------------------------------------

st.set_page_config(page_title="Anu.AI Chat", page_icon="🤖", layout="wide")

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
    role = "👤" if message["role"] == "user" else "🤖"
    st.markdown(f"**{role} {message['role'].title()}**: {message['content']}")

# Chat Input
user_input = st.text_input("💬 Type your message here...", key="chat_input")

# Send Button
if st.button("📤 Send", key="send_button", use_container_width=True):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = requests.post("http://127.0.0.1:8000/chat/", json={"message": user_input})
        bot_response = response.json().get("response", "I didn't understand that.")
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        st.markdown(f"**🤖 Anu.AI:** {bot_response}")

# 🎙️ **Voice Input Using JavaScript (Web Speech API)**
speech_text = streamlit_js_eval(js_expressions="window.navigator.mediaDevices.getUserMedia({ audio: true });", key="speech_recognition")

if speech_text:
    st.text_input("You said:", speech_text, key="user_speech")
    st.session_state.chat_history.append({"role": "user", "content": speech_text})
