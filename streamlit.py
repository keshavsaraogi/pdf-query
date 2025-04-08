import os
import cassio
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_KEY")

llm = OpenAI(openai_api_key=OPENAI_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

st.title("📄 Chat with your PDF")
st.markdown("Upload a PDF and start chatting with it using AI!")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if text:
        st.success("✅ PDF uploaded and text extracted!")

        cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)

        vectordb = Cassandra(
            embedding=embedding,
            table_name="pdf_chatbot_table",
            session=None
        )

        index = VectorStoreIndexWrapper(vectorstore=vectordb)
        vectordb.add_texts([text])

        st.success("✅ Text embedded into the vector database!")