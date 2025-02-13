import os
import requests
import streamlit as st
import torch
import boto3
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
from io import BytesIO
from dotenv import load_dotenv
from groq import Groq

# ✅ Load Environment Variables
load_dotenv()

# ✅ API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
PINECONE_INDEX_NAME = "chatbot-knowledge"

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(PINECONE_INDEX_NAME)

# ✅ Initialize API Clients
groq_client = Groq(api_key=GROQ_API_KEY)
polly_client = boto3.client("polly", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name="us-east-1")

# ✅ Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Helper Functions
def get_embedding(text):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def extract_text_from_web(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return "\n".join([p.get_text() for p in soup.find_all("p")]).strip()
    except Exception:
        return ""

def process_pdf(file):
    pdf_loader = PyPDFLoader(BytesIO(file.read()))
    text_data = "\n".join([doc.page_content for doc in pdf_loader.load()])
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_text(text_data)

def chat_with_ai(user_input, knowledge_source):
    if knowledge_source == "Chat History":
        search_results = index.query(vector=get_embedding(user_input), top_k=5, include_metadata=True)
        past_conversations = "\n".join([match["metadata"]["text"] for match in search_results["matches"]])
        context = f"Context:\n{past_conversations}\n\nUser: {user_input}"
    elif knowledge_source == "PDF" and "pdf_text" in st.session_state:
        context = f"Document Knowledge:\n{st.session_state['pdf_text']}\n\nUser: {user_input}"
    elif knowledge_source == "Web URL" and "web_text" in st.session_state:
        context = f"Web Knowledge:\n{st.session_state['web_text']}\n\nUser: {user_input}"
    else:
        context = user_input

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": "You are an AI assistant."},
                  {"role": "user", "content": context}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

# ✅ Streamlit UI
st.set_page_config(page_title="AI Assistant", page_icon="🤖")

st.sidebar.title("⚙️ AI Settings")
knowledge_source = st.sidebar.radio("Select Knowledge Source", ["Chat History", "PDF", "Web URL", "AI Model"])

if knowledge_source == "PDF":
    pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        with st.spinner("Processing PDF..."):
            st.session_state["pdf_text"] = "\n".join(process_pdf(pdf_file))
        st.sidebar.success("PDF Processed!")

if knowledge_source == "Web URL":
    url = st.sidebar.text_input("Enter Website URL")
    if st.sidebar.button("Process URL") and url:
        with st.spinner("Extracting Web Data..."):
            st.session_state["web_text"] = extract_text_from_web(url)
        st.sidebar.success("Web Data Extracted!")

st.title("💬 AI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Type your message...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        bot_reply = chat_with_ai(user_input, knowledge_source)

    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
