import os
import torch
import logging
import threading
import requests
import streamlit as st
import boto3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from dotenv import load_dotenv
from groq import Groq
import uvicorn
import socket
from io import BytesIO
import base64
import time

# ---------------------- Load Environment Variables ----------------------
load_dotenv()

# ---------------------- Setup Logging ----------------------
logging.basicConfig(level=logging.INFO)

# ---------------------- API Keys & AWS Configuration ----------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PINECONE_INDEX_NAME = "ai-multimodal-chatbot"

# Validate API Keys
if not GROQ_API_KEY or not PINECONE_API_KEY or not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("❌ ERROR: Missing API keys! Check your .env file.")

# ---------------------- Initialize AWS Clients ----------------------
polly_client = boto3.client("polly", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)
transcribe_client = boto3.client("transcribe", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

# ---------------------- Initialize Pinecone ----------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME) if PINECONE_INDEX_NAME in [i.name for i in pc.list_indexes()] else None

if index is None:
    raise ValueError(f"❌ ERROR: Pinecone index '{PINECONE_INDEX_NAME}' not found.")

# ---------------------- Load Hugging Face Models ----------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@torch.no_grad()
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_models()

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
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Retrieve past chat history from Pinecone
def get_past_conversations(user_query):
    query_embedding = get_embedding(user_query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return "\n".join([match["metadata"]["text"] for match in results["matches"]])

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

    # Store in Pinecone with optimized frequency
    if len(user_input.split()) > 2:  # Avoid storing very short inputs
        index.upsert(vectors=[
            {"id": f"user-{hash(user_input)}", "values": get_embedding(user_input), "metadata": {"text": user_input, "role": "user"}},
            {"id": f"bot-{hash(response)}", "values": get_embedding(response), "metadata": {"text": response, "role": "bot"}}
        ])

    return {"response": response}

# ---------------------- AWS Polly (Text-to-Speech) ----------------------
def text_to_speech(text):
    response = polly_client.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId="Joanna")
    return response["AudioStream"].read()

# ---------------------- AWS Transcribe (Speech-to-Text) ----------------------
def speech_to_text(audio_file):
    job_name = f"transcription-{int(time.time())}"
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": audio_file},
        MediaFormat="mp3",
        LanguageCode="en-US"
    )
    
    # Wait for transcription to complete
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        if status["TranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
            break
        time.sleep(2)
    
    if status["TranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
        transcript_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        transcript_text = requests.get(transcript_url).json()["results"]["transcripts"][0]["transcript"]
        return transcript_text
    return None

# ---------------------- Start FastAPI in a Separate Thread ----------------------
def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

port = find_available_port()

def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

threading.Thread(target=run_backend, daemon=True).start()

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="ANU.AI", page_icon="🤖", layout="wide")

st.title("💬 Anu.AI Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("💬 Type your message or use voice input...")

if st.button("🎙️ Speak"):
    audio_file = st.file_uploader("Upload your voice (MP3 only)", type=["mp3"])
    if audio_file:
        transcript_text = speech_to_text(audio_file)
        if transcript_text:
            st.write(f"🗣️ You said: {transcript_text}")
            user_input = transcript_text

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    API_URL = f"http://localhost:{port}/chat/"
    response = requests.post(API_URL, json={"message": user_input}, timeout=10)
    bot_response = response.json().get("response", "I didn't understand that.")

    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    st.write(f"🤖 {bot_response}")

    if st.button("🔊 Listen"):
        audio_data = text_to_speech(bot_response)
        b64 = base64.b64encode(audio_data).decode()
        st.audio(f"data:audio/mp3;base64,{b64}", format="audio/mp3")
