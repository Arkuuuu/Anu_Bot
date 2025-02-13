import os
import torch
import requests
import logging
import streamlit as st
import numpy as np
import boto3
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

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

if not GROQ_API_KEY or not AWS_ACCESS_KEY or not AWS_SECRET_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ ERROR: Missing API keys! Check your .env file.")

# ---------------------- Initialize Services ----------------------
# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(PINECONE_INDEX_NAME, dimension=384, metric="cosine")

index = pc.Index(PINECONE_INDEX_NAME)

# ✅ Initialize AWS Polly (TTS)
polly_client = boto3.client("polly", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

# ✅ Load Hugging Face Model for Embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# ---------------------- Utility Functions ----------------------
def get_embedding(text):
    """Generate text embeddings using Hugging Face model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def get_past_conversations(user_query):
    """Retrieve past chat conversations from Pinecone."""
    query_embedding = get_embedding(user_query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return "\n".join([match["metadata"]["text"] for match in results["matches"]])

def extract_text_from_web(url):
    """Scrape text content from a webpage."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return "\n".join([p.get_text() for p in soup.find_all("p")]).strip()
    except:
        return "❌ Unable to extract text from URL."

def process_pdf(pdf_file):
    """Extract text from uploaded PDF."""
    pdf_loader = PyPDFLoader(BytesIO(pdf_file.read()))
    text_data = "\n".join([doc.page_content for doc in pdf_loader.load()])
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_text(text_data)

def text_to_speech(text):
    """Convert bot response to speech using AWS Polly."""
    response = polly_client.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId="Joanna")
    audio_stream = response["AudioStream"].read()
    return audio_stream

def chat_with_ai(user_input, knowledge_source):
    """Generate AI response based on selected knowledge source."""
    if knowledge_source == "Chat History":
        context = f"Previous Conversations:\n{get_past_conversations(user_input)}\n\nUser: {user_input}"
    elif knowledge_source == "PDF" and "pdf_text" in st.session_state:
        context = f"Document Data:\n{st.session_state['pdf_text']}\n\nUser: {user_input}"
    elif knowledge_source == "Web URL" and "web_text" in st.session_state:
        context = f"Web Data:\n{st.session_state['web_text']}\n\nUser: {user_input}"
    else:
        context = user_input

    # Get AI response
    response = requests.post("https://api.groq.com/v1/chat/completions", json={
        "messages": [{"role": "system", "content": "You are an AI assistant."},
                     {"role": "user", "content": context}],
        "model": "llama-3.3-70b-versatile"
    }, headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
    
    return response.json()["choices"][0]["message"]["content"]

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Anu AI Chatbot", page_icon="🤖", layout="wide")

# Sidebar Configuration
st.sidebar.title("⚙️ AI Settings")
knowledge_source = st.sidebar.radio("Select Knowledge Source", ["Chat History", "PDF", "Web URL", "AI Model"])

# 📄 PDF Upload
if knowledge_source == "PDF":
    pdf_file = st.sidebar.file_uploader("📄 Upload PDF", type=["pdf"])
    if pdf_file:
        with st.spinner("Processing PDF..."):
            st.session_state["pdf_text"] = "\n".join(process_pdf(pdf_file))
        st.sidebar.success("✅ PDF Processed!")

# 🌐 Web URL Processing
if knowledge_source == "Web URL":
    url = st.sidebar.text_input("🌐 Enter Website URL")
    if st.sidebar.button("🔍 Process URL") and url:
        with st.spinner("Extracting Web Data..."):
            st.session_state["web_text"] = extract_text_from_web(url)
        st.sidebar.success("✅ Web Data Extracted!")

# Chat Title
st.title("💬 Anu AI Chatbot")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input & AI Response
user_input = st.chat_input("Type your message...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking... 🤖"):
        bot_reply = chat_with_ai(user_input, knowledge_source)

    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

# 🔊 TTS Button
if "last_response" in st.session_state:
    if st.sidebar.button("🔊 Listen to AI Response"):
        audio_data = text_to_speech(st.session_state["last_response"])
        st.audio(audio_data, format="audio/mp3")

st.session_state["last_response"] = bot_reply
