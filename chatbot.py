import os
import torch
import requests
import logging
import asyncio
import streamlit as st
import numpy as np
import boto3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from transformers import AutoTokenizer, AutoModel, ViTFeatureExtractor, ViTForImageClassification
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from bs4 import BeautifulSoup


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
polly_client = boto3.client(
    "polly",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Initialize FastAPI
app = FastAPI()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX_NAME)

# Load Hugging Face Models
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Update Vision Transformer (ViT) to use `AutoImageProcessor`
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

# Function to Handle Chat Requests
@app.post("/chat/")
async def chat(query: dict):
    user_input = query.get("message", "")
    image_url = query.get("image_url", None)

    context = ""
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = vit_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            class_id = torch.argmax(predictions, dim=-1).item()
            context = f"Image contains: {vit_model.config.id2label[class_id]}\n"

    try:
        query_embedding = get_embedding(user_input)
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        retrieved_text = "\n".join([r["metadata"]["text"] for r in results["matches"]])
    except Exception:
        retrieved_text = ""

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": "You are a vision-aware AI assistant."},
                  {"role": "user", "content": f"Context:\n{context}{retrieved_text}\nQuestion: {user_input}"}],
        model="llama-3.3-70b-versatile"
    )

    return {"response": chat_completion.choices[0].message.content}

# WebSocket for STT
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
                response = requests.post("https://api.groq.com/audio/transcriptions", headers=headers, files=files)
                transcription = response.json().get("text", "")
                await websocket.send_text(transcription)
                audio_buffer.clear()
    except WebSocketDisconnect:
        logging.warning("STT WebSocket disconnected.")
    finally:
        await websocket.close()

# WebSocket for TTS
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

# ----------------- STREAMLIT UI -----------------
# Backend URLs
backend_url = "http://localhost:8000/chat/"
image_api_url = "http://localhost:8000/analyze_image/"
stt_ws_url = "ws://localhost:8000/stt/"

# Set Page Configurations
st.set_page_config(page_title="ANU.AI", page_icon="🤖", layout="wide")

# Custom CSS for Transitions & Styling
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message-container {
            animation: fadeIn 0.5s ease-out;
        }
        .chatbox {
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .chat-user {
            background: #2b6cb0;
            color: white;
            text-align: right;
        }
        .chat-bot {
            background: #1a202c;
            color: white;
        }
        .center {
            text-align: center;
        }
        .responsive {
            max-width: 100%;
            width: 90%;
            margin: auto;
        }
        .button {
            transition: all 0.3s ease;
        }
        .button:hover {
            transform: scale(1.05);
        }
        .recording {
            animation: blink 1s infinite alternate;
        }
        @keyframes blink {
            from { background: red; }
            to { background: #e53e3e; }
        }
    </style>
""", unsafe_allow_html=True)

# Header with Icon & Name
st.markdown("<h1 class='center'>🤖 ANU.AI - Your Intelligent Assistant</h1>", unsafe_allow_html=True)

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Image Upload Section
uploaded_file = st.file_uploader("📸 Upload an image for analysis", type=["jpg", "png"])
if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    response = requests.post(image_api_url, files={"file": image_bytes})
    prediction = response.json().get("prediction", "Unknown")
    st.image(uploaded_file, caption=f"Detected: {prediction}", use_column_width=True)

# Display Chat Messages
for message in st.session_state.chat_history:
    role_class = "chat-user" if message["role"] == "user" else "chat-bot"
    st.markdown(f"<div class='chatbox {role_class} message-container'>{message['content']}</div>", unsafe_allow_html=True)

# Speech-to-Text (STT) WebSocket Connection
def speech_to_text():
    ws = websocket.WebSocket()
    ws.connect(stt_ws_url)
    
    with st.spinner("🎙️ Listening... Speak now!"):
        sleep(5)  # Simulate 5-second speech recording
        ws.send(b"test audio data")  # Placeholder for real audio data
        transcript = ws.recv()  # Receive transcribed text
        ws.close()
    
    st.session_state.chat_input = transcript  # Auto-fill chat input

# Chat Input
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input("💬 Type your message here...", key="chat_input")

with col2:
    if st.button("🎙️", help="Click to speak", key="mic_button", use_container_width=True, css_classes=["button", "recording"]):
        threading.Thread(target=speech_to_text).start()

# Send Button
if st.button("Send", help="Click to send", key="send_button", use_container_width=True, css_classes=["button"]):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.markdown(f"<div class='chatbox chat-user message-container'>{user_input}</div>", unsafe_allow_html=True)
        
        with st.spinner("🤖 ANU.AI is thinking..."):
            sleep(1)  # Simulate processing delay
            response = requests.post(backend_url, json={"message": user_input})
            bot_response = response.json().get("response", "I didn't understand that.")
        
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        st.markdown(f"<div class='chatbox chat-bot message-container'>{bot_response}</div>", unsafe_allow_html=True)
