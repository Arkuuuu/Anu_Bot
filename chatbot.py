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
    raise ValueError("❌ ERROR: Missing API keys! Please check your .env file.")

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
    raise ValueError(f"❌ ERROR: Pinecone index '{PINECONE_INDEX_NAME}' not found.")

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

# ---------------------- Clear Chat on Refresh ----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat Messages
st.title("💬 Anu.AI Chat")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------- Custom Input UI ----------------------
st.markdown("""
    <style>
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
            z-index: 1000;
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
            width: 70%;
        }
        .btn {
            width: 45px;
            height: 45px;
            margin-left: 8px;
            border-radius: 50%;
            background: #30363d;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 22px;
            color: white;
            border: none;
        }
        .btn:hover {
            background: #484f58;
        }
    </style>
""", unsafe_allow_html=True)

st.components.v1.html("""
    <div class="chat-container">
        <button id="mic-btn" class="btn">🎙️</button>
        <input id="chat-input" class="chat-input" type="text" placeholder="Type your message...">
        <button id="send-btn" class="btn">📤</button>
    </div>
    <script>
        document.getElementById("send-btn").addEventListener("click", function() {
            var userMessage = document.getElementById("chat-input").value;
            if (userMessage.trim() !== "") {
                window.parent.postMessage({ type: "user_message", message: userMessage }, "*");
                document.getElementById("chat-input").value = "";
            }
        });

        document.getElementById("mic-btn").addEventListener("click", function() {
            window.parent.postMessage({ type: "voice_input" }, "*");
        });
    </script>
""", height=100)

# Start FastAPI inside Streamlit
def run_fastapi():
    run(app, host="0.0.0.0", port=8000, log_level="info")

# Run FastAPI in a separate thread
threading.Thread(target=run_fastapi, daemon=True).start()
