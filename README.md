# RAG Chatbot Over Real Documents

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on information retrieved from uploaded documents.

## Features
- Extracts text from documents
- Splits text into smaller chunks
- Retrieves relevant information for user queries
- Generates answers using retrieved context
- Supports document-based question answering

## Tech Stack
- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- PyPDF

## Project Structure
- `app.py` — Main application
- `college_handbook.txt` — Sample document
- `requirements.txt` — Python dependencies

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/nidhi3106-ai/Rag-chatbot.git
   ```

2. Open the project folder:
   ```bash
   cd Rag-chatbot
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Learning Goals
This project demonstrates document processing, text chunking, semantic search, vector databases, and Retrieval-Augmented Generation.

## Author
Nidhi Mundhada
