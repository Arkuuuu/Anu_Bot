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
    raise ValueError("\u274c ERROR: Missing API keys! Please check your .env file or Streamlit secrets.")

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)
polly_client = boto3.client("polly", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
available_indexes = [index.name for index in pc.list_indexes()]

if PINECONE_INDEX_NAME in available_indexes:
    logging.info(f"\u2705 Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    index = pc.Index(PINECONE_INDEX_NAME)
else:
    raise ValueError(f"\u274c ERROR: Pinecone index '{PINECONE_INDEX_NAME}' not found. Check your Pinecone dashboard.")

# Load Hugging Face Models
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Updated ViT Model for Image Processing
vit_model_name = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(vit_model_name)
vit_model = ViTForImageClassification.from_pretrained(vit_model_name)

# ------------------------- FastAPI Backend -------------------------
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
    past_conversations = [match["metadata"]["text"] for match in results["matches"]]
    return "\n".join(past_conversations)

@app.post("/chat/")
async def chat(query: dict):
    user_input = query.get("message", "")
    past_chats = get_past_conversations(user_input)

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": f"Previous Context:\n{past_chats}\nUser: {user_input}"}
        ],
        model="llama-3.3-70b-versatile"
    )

    response = chat_completion.choices[0].message.content

    index.upsert(vectors=[
        {"id": f"user-{hash(user_input)}", "values": get_embedding(user_input), "metadata": {"text": user_input, "role": "user"}},
        {"id": f"bot-{hash(response)}", "values": get_embedding(response), "metadata": {"text": response, "role": "bot"}}
    ])

    return {"response": response}

# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="Anu.AI Chat", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
        body { background-color: #0F172A; color: white; }
        .stTextInput>div>div>input { background: #1E293B; color: white; }
        .stButton>button { background: #2563EB; color: white; border-radius: 5px; }
        .chat-bubble { padding: 10px; border-radius: 10px; max-width: 60%; }
        .chat-user { background: #2563EB; color: white; text-align: right; }
        .chat-bot { background: #1E293B; color: white; text-align: left; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>🤖 Anu.AI Chat</h2>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role_class = "chat-user" if message["role"] == "user" else "chat-bot"
    st.markdown(f"<div class='chat-bubble {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

user_input = st.text_input("💬 Type your message here...", key="chat_input")
if st.button("📤 Send"):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = requests.post("http://127.0.0.1:8000/chat/", json={"message": user_input})
        bot_response = response.json().get("response", "I didn't understand that.")
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        st.markdown(f"<div class='chat-bubble chat-bot'>{bot_response}</div>", unsafe_allow_html=True)

def run_fastapi():
    run(app, host="0.0.0.0", port=8000, log_level="info")

threading.Thread(target=run_fastapi, daemon=True).start()
