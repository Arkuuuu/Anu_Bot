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
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PINECONE_INDEX_NAME = "ai-multimodal-chatbot"

# Validate API Keys
if not GROQ_API_KEY or not AWS_ACCESS_KEY or not AWS_SECRET_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ ERROR: Missing API keys! Please check your .env file or Streamlit secrets.")

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)
polly_client = boto3.client("polly", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

# Initialize FastAPI
app = FastAPI()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the Pinecone index exists before creating
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

# -----------------------------------------------------------------Streamlit UI---------------------------------------------------------------
st.set_page_config(page_title="Anu.AI Chat", page_icon="🤖", layout="wide")

# Custom CSS Styling for Dark Mode UI
st.markdown("""
    <style>
        /* Background and Text */
        body {
            background-color: #1a1b26;
            color: white;
        }
        .main {
            background-color: #1a1b26;
        }
        
        /* Chat UI */
        .chat-container {
            padding: 20px;
        }
        .chat-bubble {
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 60%;
            display: inline-block;
            margin-bottom: 10px;
        }
        .chat-user {
            background-color: #2b6cb0;
            color: white;
            text-align: right;
            margin-left: auto;
        }
        .chat-bot {
            background-color: #1e293b;
            color: white;
            text-align: left;
        }
        
        /* Input box */
        .chat-input {
            width: 90%;
            padding: 10px;
            border-radius: 10px;
            border: none;
            background: #2d2f3b;
            color: white;
        }

        /* Sidebar Styling */
        .sidebar {
            background-color: #1a1b26;
            padding: 15px;
        }
        .sidebar button {
            width: 100%;
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 10px;
            background: #2b6cb0;
            color: white;
            border: none;
        }

        /* Floating action buttons */
        .action-btn {
            background: #2b6cb0;
            color: white;
            border: none;
            padding: 10px;
            margin-right: 5px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Chat Header
st.markdown("<h2 style='text-align: center;'>🤖 Anu.AI Chat</h2>", unsafe_allow_html=True)

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

# Display Chat Messages with Styled Bubbles
for message in st.session_state.chat_history:
    role_class = "chat-user" if message["role"] == "user" else "chat-bot"
    st.markdown(f"<div class='chat-bubble {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

# Chat Input Box
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input("Type your message...", key="chat_input")

# Send Button
if st.button("📤 Send", key="send_button", use_container_width=True):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = requests.post("http://localhost:8000/chat/", json={"message": user_input})
        bot_response = response.json().get("response", "I didn't understand that.")
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        st.markdown(f"<div class='chat-bubble chat-bot'>{bot_response}</div>", unsafe_allow_html=True)

# Image Upload for OCR & Object Detection
uploaded_file = st.file_uploader("📸 Upload an image for analysis", type=["jpg", "png"])
if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    response = requests.post("http://localhost:8000/analyze_image/", files={"file": image_bytes})
    result = response.json()
    st.image(uploaded_file, caption=f"🖼 Detected: {result['prediction']}", use_column_width=True)
    st.markdown(f"📝 Extracted Text: `{result['extracted_text']}`")

# 🎙️ **Voice Input Using JavaScript (Web Speech API)**
st.write("🎙️ Click below to use voice input:")
speech_text = streamlit_js_eval(js_expressions="window.navigator.mediaDevices.getUserMedia({ audio: true });", key="speech_recognition")

if speech_text:
    st.text_input("You said:", speech_text, key="user_speech")
    st.session_state.chat_history.append({"role": "user", "content": speech_text})
