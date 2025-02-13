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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, ViTForImageClassification
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

# Updated ViT Model for Image Processing
vit_model_name = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(vit_model_name)
vit_model = ViTForImageClassification.from_pretrained(vit_model_name)

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

# Function to Generate Embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.tolist()

# Store and Retrieve Chat History from Pinecone
def get_past_conversations(user_query):
    query_embedding = get_embedding(user_query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    past_conversations = []
    for match in results["matches"]:
        past_conversations.append(match["metadata"]["text"])

    return "\n".join(past_conversations)

@app.post("/chat/")
async def chat(query: dict):
    user_input = query.get("message", "")

    # Retrieve past chats from Pinecone
    past_chats = get_past_conversations(user_input)

    # Get AI response
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": f"Previous Context:\n{past_chats}\nUser: {user_input}"}
        ],
        model="llama-3.3-70b-versatile"
    )

    response = chat_completion.choices[0].message.content

    # Store both in Pinecone
    index.upsert(vectors=[
        {"id": f"user-{hash(user_input)}", "values": get_embedding(user_input), "metadata": {"text": user_input, "role": "user"}},
        {"id": f"bot-{hash(response)}", "values": get_embedding(response), "metadata": {"text": response, "role": "bot"}}
    ])

    return {"response": response}

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="ANU.AI", page_icon="🤖", layout="wide")

# Sidebar for Quick Actions
with st.sidebar:
    st.title("Settings")
    st.button("📥 Download Chat")
    st.button("🔗 Share Chat")
    st.subheader("Quick Actions")
    st.button("💻 Help me write code")
    st.button("📖 Explain a concept")
    st.button("🎨 Generate ideas")

# Fetch chat history from Pinecone
def fetch_chat_history():
    results = index.query(vector=get_embedding("recent chats"), top_k=10, include_metadata=True)
    return [{"role": match["metadata"]["role"], "content": match["metadata"]["text"]} for match in results["matches"]]

# Load chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = fetch_chat_history()

# Display Chat Messages
st.title("💬 Anu.AI Chat")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 🎙️ **Voice Input Using JavaScript (Web Speech API)**
st.write("🎙️ Click the mic button to use voice input:")
col1, col2 = st.columns([8, 1])

with col1:
    user_input = st.chat_input("💬 Type your message here...")

with col2:
    mic_clicked = st.button("🎙️", key="mic_button")

if mic_clicked:
    speech_text = streamlit_js_eval(js_expressions="window.navigator.mediaDevices.getUserMedia({ audio: true });", key="speech_recognition")
    
    if speech_text:
        st.session_state.chat_history.append({"role": "user", "content": speech_text})
        try:
            response = requests.post("http://127.0.0.1:8000/chat/", json={"message": speech_text}, timeout=10)
            bot_response = response.json().get("response", "I didn't understand that.")
        except requests.exceptions.RequestException as e:
            bot_response = f"⚠️ Error: {str(e)}"

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.write(bot_response)

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    try:
        response = requests.post("http://127.0.0.1:8000/chat/", json={"message": user_input}, timeout=10)
        bot_response = response.json().get("response", "I didn't understand that.")
    except requests.exceptions.RequestException as e:
        bot_response = f"⚠️ Error: {str(e)}"

    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.write(bot_response)

# Start FastAPI inside Streamlit
def run_fastapi():
    run(app, host="0.0.0.0", port=8000, log_level="info")

# Run FastAPI in a separate thread
threading.Thread(target=run_fastapi, daemon=True).start()
