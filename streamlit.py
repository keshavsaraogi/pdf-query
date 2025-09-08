import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI, OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.markdown("Upload a PDF and start chatting with it using AI!")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_KEY")
    if missing:
        st.error(
            "Missing required environment variables: " + ", ".join(missing) +
            ". Please set them in your .env file."
        )
        st.stop()

    pdf_reader = PdfReader(uploaded_file)
    
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if text:
        st.success("✅ PDF uploaded and text extracted!")

        vectordb = FAISS.from_texts([text], embedding=embedding)
        index = VectorStoreIndexWrapper(vectorstore=vectordb)

        st.success("✅ Text embedded into the FAISS index!")
        st.subheader("Ask a question about your PDF 👇")

        if "history" not in st.session_state:
            st.session_state.history = []

        user_question = st.text_input("Your question:")
        if user_question:
            response = index.query(user_question, llm=llm)
            st.session_state.history.append((user_question, response))

        for user_q, bot_a in st.session_state.history:
            with st.chat_message("user"):
                st.write(user_q)
            with st.chat_message("assistant"):
                st.write(bot_a)

    else:
        st.error("❌ Could not extract any text from the PDF. Please try another file.")
else:
    st.info("👆 Upload a PDF file to get started!")
