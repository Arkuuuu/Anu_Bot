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

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the Pinecone index exists before using it
if PINECONE_INDEX_NAME in pc.list_indexes():
    logging.info(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    index = pc.Index(PINECONE_INDEX_NAME)
else:
    raise ValueError(f"❌ ERROR: Pinecone index '{PINECONE_INDEX_NAME}' not found. Check your Pinecone dashboard.")


# Load Hugging Face Models
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Updated ViT Model for Image Processing
vit_model_name = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(vit_model_name)
vit_model = ViTForImageClassification.from_pretrained(vit_model_name)

# Function to Generate Embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.tolist()

# OCR Function for Text Extraction from Images
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text.strip()

# Function to Process Images
@app.post("/analyze_image/")
async def analyze_image(file: UploadFile):
    image = Image.open(BytesIO(await file.read()))
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        class_id = torch.argmax(predictions, dim=-1).item()
    
    extracted_text = extract_text_from_image(image)

    return {
        "prediction": vit_model.config.id2label[class_id],
        "extracted_text": extracted_text
    }

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
    except WebSocketDisconnect:
        logging.warning("TTS WebSocket disconnected.")
    finally:
        await websocket.close()

# -----------------------------------------------------------------Streamlit UI---------------------------------------------------------------
st.set_page_config(page_title="ANU.AI", page_icon="🤖", layout="wide")

# Chat Header
st.markdown("<h1 style='text-align: center;'>🤖 ANU.AI - Your Smart AI Assistant</h1>", unsafe_allow_html=True)

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat Messages
for message in st.session_state.chat_history:
    role = "👤" if message["role"] == "user" else "🤖"
    st.markdown(f"**{role} {message['role'].title()}**: {message['content']}")

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
