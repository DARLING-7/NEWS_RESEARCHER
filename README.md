# NEWS_RESEARCHER
simply news searching 

# News Research Tool

## Overview
The **News Research Tool** is a Streamlit-based application that allows users to input URLs of news articles, process their content, and perform question-answering using Google's Gemini AI model. The tool leverages LangChain, FAISS for vector storage, and sentence embeddings for efficient retrieval-based question answering.

## Features
- Extracts text from given URLs using `UnstructuredURLLoader`.
- Splits extracted text into smaller chunks for better processing.
- Generates embeddings using `SentenceTransformerEmbeddings`.
- Stores and retrieves embeddings efficiently using FAISS.
- Answers user queries using the Gemini AI model via LangChain’s `RetrievalQAWithSourcesChain`.
- Provides cited sources for answers where available.

## Tech Stack
- **Python** (Core programming language)
- **Streamlit** (User interface)
- **LangChain** (For LLM-based workflows)
- **Google Gemini AI** (LLM for text generation)
- **FAISS** (Vector database for retrieval)
- **Sentence-Transformers** (For embeddings)
- **Unstructured** (For web scraping)
- **Pickle** (For saving vectorstore)

## Installation
### Prerequisites
Ensure you have Python installed (>=3.8) and `pip` package manager.

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/news-research-tool.git
   cd news-research-tool
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up your API key for Google Gemini AI:
   - Create a `.env` file in the project root.
   - Add your API key:
     ```sh
     GEMINI_API_KEY=your_api_key_here
     ```
4. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Enter up to 3 article URLs in the sidebar.
2. Click **Process URLs** to fetch and process content.
3. Ask a question in the input field to retrieve an answer.
4. View the answer and its sources in the main display area.

## File Structure
```
news-research-tool/
│── app.py               # Main Streamlit application
│── .env                 # Environment variables (API keys)
│── requirements.txt     # Dependencies
│── faiss_store_gemini.pkl  # FAISS vectorstore (saved)
│── README.md            # Documentation
```

## Future Improvements
- Support for more than 3 URLs.
- Enhanced UI/UX with better query experience.
- Alternative LLM support (e.g., OpenAI, Ollama).
- Summarization of articles before Q&A.
- Multi-document comparison and analysis.

## License
This project is open-source and available under the [MIT License](LICENSE).

---
**Developed by:** [JAGADEESH KATTA]
