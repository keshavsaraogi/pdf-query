
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize Astra DB and OpenAI
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_KEY")

# Setup OpenAI
llm = OpenAI(openai_api_key=OPENAI_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

# Streamlit UI
st.title("📄 Chat with your PDF")