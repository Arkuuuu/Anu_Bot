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

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)
polly_client = boto3.client("polly", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

# Initialize FastAPI
app = FastAPI()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes():
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(PINECONE_INDEX_NAME)

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
st.markdown("<h1 class='center'>🤖 ANU.AI - Your Smart AI Assistant</h1>", unsafe_allow_html=True)

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat Messages
for message in st.session_state.chat_history:
    role = "👤" if message["role"] == "user" else "🤖"
    st.markdown(f"**{role} {message['role'].title()}**: {message['content']}")

# 🎙️ **Voice Input Using JavaScript (Web Speech API)**
st.write("🎙️ Click below to use voice input:")

speech_text = streamlit_js_eval(js_expressions="window.navigator.mediaDevices.getUserMedia({ audio: true });", key="speech_recognition")

if speech_text:
    st.text_input("You said:", speech_text, key="user_input")
    st.session_state.chat_history.append({"role": "user", "content": speech_text})

    # Send to chatbot backend
    response = requests.post("http://localhost:8000/chat/", json={"message": speech_text})
    bot_response = response.json().get("response", "I didn't understand that.")

    # Display bot response
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    st.markdown(f"**🤖 ANU.AI:** {bot_response}")

# Image Upload for OCR & Object Detection
uploaded_file = st.file_uploader("📸 Upload an image for analysis", type=["jpg", "png"])
if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    response = requests.post("http://localhost:8000/analyze_image/", files={"file": image_bytes})
    result = response.json()
    st.image(uploaded_file, caption=f"Detected: {result['prediction']}", use_column_width=True)
    st.markdown(f"📝 Extracted Text: `{result['extracted_text']}`")

# Chat Input with Emoji Support
user_input = st.text_input("💬 Type your message here...", key="chat_input")

emojis = ["😊", "👍", "🎉", "🤔", "💡", "✨", "🚀", "💪"]
selected_emoji = st.radio("Pick an emoji to react", emojis, horizontal=True)
if selected_emoji:
    st.session_state.chat_history.append({"role": "user", "content": selected_emoji})

# Send Button
if st.button("Send", key="send_button", use_container_width=True):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = requests.post("http://localhost:8000/chat/", json={"message": user_input})
        bot_response = response.json().get("response", "I didn't understand that.")
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        st.markdown(f"**🤖 ANU.AI:** {bot_response}")
