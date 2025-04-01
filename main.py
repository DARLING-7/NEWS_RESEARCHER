import os
import streamlit as st
import pickle
import time
import google.generativeai as genai
from langchain.llms.base import LLM
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from typing import Optional, List

# Load environment variables from .env (e.g., GEMINI_API_KEY)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Custom Gemini LLM class for LangChain compatibility
class GeminiLLM(LLM):
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.9
    max_tokens: int = 500
    _model_instance = None  # Private attribute to hold the model instance

    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.9, max_tokens: int = 500):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model_instance = genai.GenerativeModel(model_name)  # Store model instance privately

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._model_instance.generate_content(
            contents=prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }
        )
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"

# Streamlit UI
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini.pkl"

main_placeholder = st.empty()

# Initialize Gemini AI model
llm = GeminiLLM(model_name="gemini-1.5-flash", temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # Create embeddings and save to FAISS index
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore_gemini = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_gemini, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # Result will be a dictionary like {"answer": "", "sources": []}
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
