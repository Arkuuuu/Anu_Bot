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
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, ViTForImageClassification
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from groq import Groq
from time import sleep

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

# Updated ViT Model
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

# Function to Process Images
@app.post("/analyze_image/")
async def analyze_image(file: UploadFile):
    image = Image.open(BytesIO(await file.read()))
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        class_id = torch.argmax(predictions, dim=-1).item()
    return {"prediction": vit_model.config.id2label[class_id]}

# Speech-to-Text (STT) WebSocket Connection
def record_audio(duration=5, samplerate=16000):
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    return recording.tobytes()  

def speech_to_text():
    ws = websocket.WebSocket()
    ws.connect("ws://localhost:8000/stt/")
    
    with st.spinner("🎙️ Listening... Speak now!"):
        audio_data = record_audio(duration=5)
        ws.send(audio_data)  
        transcript = ws.recv()
        ws.close()
    
    st.session_state.chat_input = transcript

# Streamlit UI
st.set_page_config(page_title="ANU.AI", page_icon="🤖", layout="wide")
st.markdown("<h1 class='center'>🤖 ANU.AI - Your Smart AI Assistant</h1>", unsafe_allow_html=True)

# Chat Input
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input("💬 Type your message here...", key="chat_input")

with col2:
    if st.button("🎙️", key="mic_button", use_container_width=True):
        threading.Thread(target=speech_to_text).start()

# Send Button
if st.button("Send", key="send_button", use_container_width=True):
    if user_input:
        response = requests.post("http://localhost:8000/chat/", json={"message": user_input})
        bot_response = response.json().get("response", "I didn't understand that.")
        st.markdown(f"**🧠 ANU.AI:** {bot_response}")
