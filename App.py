import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Hardcode your Google API key here (replace with your actual key)


def get_pdf_text(pdf_docs):
    """Extract text from PDF files"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        except Exception as e:
            st.error(f"Error processing PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks if chunks else []
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return []

def get_vector_store(text_chunks, api_key):
    """Create and save vector store"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        # Test embedding with a small sample
        embeddings.embed_query("test")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def get_conversational_chain(api_key):
    """Create conversational chain"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say "The answer is not available in the context".
    Do not provide incorrect information.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """

    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

def user_input(user_question, api_key):
    """Process user question and generate response"""
    try:
        if not os.path.exists("faiss_index"):
            st.error("No FAISS index found. Please upload and process PDF files first.")
            return

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question, k=4)

        chain = get_conversational_chain(api_key)
        if chain is None:
            return

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    """Main application function"""
    st.set_page_config(page_title="Chat PDF", layout="wide")
    
    api_key = GOOGLE_API_KEY

    if not api_key or api_key == "your-actual-api-key-here":
        st.error("Please set a valid GOOGLE_API_KEY in the code.")
        st.info("Replace 'your-actual-api-key-here' with your actual Google API key at the top of the script.")
        return
    
    # Display partial API key for debugging
    st.sidebar.write(f"Using API Key (first 5 chars): {api_key[:5]}...")

    try:
        genai.configure(api_key=api_key)
        # Test API key with a simple call
        genai.get_model("models/gemini-1.5-flash")
    except Exception as e:
        st.error(f"Error validating API key: {str(e)}")
        st.info("Please verify your API key in Google Cloud Console and ensure Generative Language API is enabled.")
        return

    st.header("Chat with PDF using Gemini💁")

    if 'processed' not in st.session_state:
        st.session_state.processed = False

    user_question = st.text_input("Ask a Question from the PDF Files", key="question_input")
    if user_question and st.session_state.processed:
        user_input(user_question, api_key)
    elif user_question and not st.session_state.processed:
        st.warning("Please upload and process PDF files first")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            type=['pdf'],
            key="pdf_uploader"
        )
        
        if st.button("Submit & Process", key="process_button"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from the PDFs")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("Failed to split text into chunks")
                        elif get_vector_store(text_chunks, api_key):
                            st.session_state.processed = True
                            st.success("Processing complete! You can now ask questions.")
                        else:
                            st.error("Failed to create vector store")

if __name__ == "__main__":
    main()