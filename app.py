
import os
import io
import hashlib

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# =========================================
# 1. PAGE CONFIGURATION
# =========================================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

# =========================================
# 2. LOAD MODELS AND DATABASE
# =========================================
@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="multi_document_rag",
        metadata={"hnsw:space": "cosine"}
    )

    return embedding_model, client, collection


embedding_model, chroma_client, collection = load_resources()

# =========================================
# 3. SESSION STATE
# =========================================
defaults = {
    "messages": [],
    "document_id": None,
    "document_name": None,
    "upload_message": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================
# 4. EXTRACT TEXT FROM PDF OR TXT
# =========================================
def extract_text(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        text = "\n".join(pages)

    elif uploaded_file.name.lower().endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError("Please upload a PDF or TXT file.")

    if not text.strip():
        raise ValueError(
            f"No readable text found in {uploaded_file.name}. "
            "Scanned PDFs may require OCR."
        )

    return text

# =========================================
# 5. CHUNKING
# =========================================
def create_chunks(text, chunk_size=700, overlap=100):
    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks

# =========================================
# 6. STORE MULTIPLE DOCUMENTS
# =========================================
def store_documents(documents):
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    all_chunks = []
    all_ids = []
    all_metadata = []

    for doc_index, document in enumerate(documents):
        for chunk_index, chunk in enumerate(document["chunks"]):
            all_chunks.append(chunk)
            all_ids.append(f"doc{doc_index}_chunk{chunk_index}")

            all_metadata.append({
                "source": document["name"],
                "chunk_number": chunk_index + 1
            })

    if not all_chunks:
        raise ValueError("No text chunks were created.")

    batch_size = 100

    for start in range(0, len(all_chunks), batch_size):
        end = start + batch_size

        chunks_batch = all_chunks[start:end]
        ids_batch = all_ids[start:end]
        metadata_batch = all_metadata[start:end]

        embeddings = embedding_model.encode(
            chunks_batch,
            normalize_embeddings=True
        ).tolist()

        collection.add(
            ids=ids_batch,
            documents=chunks_batch,
            embeddings=embeddings,
            metadatas=metadata_batch
        )

# =========================================
# 7. SIDEBAR
# =========================================
st.sidebar.title("📚 RAG Chatbot")
st.sidebar.write("Upload multiple documents to ask questions.")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.session_state.upload_message:
    st.sidebar.success(st.session_state.upload_message)
    st.session_state.upload_message = None

# =========================================
# 8. PROCESS UPLOADED FILES
# =========================================
if uploaded_files:
    file_hashes = []

    for file in uploaded_files:
        file_hash = hashlib.sha256(
            file.getvalue()
        ).hexdigest()

        file_hashes.append(f"{file.name}:{file_hash}")

    new_document_id = hashlib.sha256(
        "|".join(sorted(file_hashes)).encode("utf-8")
    ).hexdigest()

    st.sidebar.info(f"{len(uploaded_files)} file(s) selected.")

    if new_document_id != st.session_state.document_id:
        try:
            with st.spinner("Processing your documents..."):
                documents = []

                for file in uploaded_files:
                    text = extract_text(file)
                    chunks = create_chunks(text)

                    documents.append({
                        "name": file.name,
                        "chunks": chunks
                    })

                store_documents(documents)

                total_chunks = sum(
                    len(doc["chunks"]) for doc in documents
                )

                st.session_state.document_id = new_document_id
                st.session_state.document_name = ", ".join(
                    doc["name"] for doc in documents
                )
                st.session_state.messages = []

                st.session_state.upload_message = (
                    f"{len(documents)} document(s) processed "
                    f"successfully! {total_chunks} chunks created."
                )

            st.rerun()

        except Exception as error:
            st.sidebar.error(f"Document processing failed: {error}")

# =========================================
# 9. DEFAULT HANDBOOK
# =========================================
elif st.session_state.document_id is None:
    if os.path.exists("college_handbook.txt"):
        try:
            with open(
                "college_handbook.txt",
                "r",
                encoding="utf-8"
            ) as file:
                text = file.read()

            chunks = create_chunks(text)

            store_documents([{
                "name": "college_handbook.txt",
                "chunks": chunks
            }])

            st.session_state.document_id = "default_handbook"
            st.session_state.document_name = "college_handbook.txt"

        except Exception as error:
            st.sidebar.error(f"Default document error: {error}")

# =========================================
# 10. ACTIVE DOCUMENTS AND CLEAR CHAT
# =========================================
if st.session_state.document_name:
    st.sidebar.caption("Active document(s):")
    st.sidebar.write(st.session_state.document_name)

if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# =========================================
# 11. MAIN PAGE
# =========================================
st.title("📚 Chat with Multiple Documents")

st.write(
    "Upload PDFs or TXT files and ask questions about their contents."
)

st.info(
    "Tip: Upload related documents together for better answers "
    "across your document collection."
)

with st.expander("💡 Example questions"):
    st.write("- What is the attendance requirement?")
    st.write("- Summarize the important points in my documents.")
    st.write("- Which document mentions examination rules?")

# =========================================
# 12. CHAT HISTORY
# =========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📖 View sources"):
                for source in message["sources"]:
                    st.write(f"File: {source['name']}")
                    st.write(source["text"])

# =========================================
# 13. ASK QUESTIONS
# =========================================
question = st.chat_input("Ask about your documents...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            answer = (
                "OPENAI_API_KEY not found. Check your .env file."
            )
            st.error(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        elif collection.count() == 0:
            answer = "Please upload a document first."
            st.warning(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        else:
            try:
                with st.spinner("Searching your documents..."):
                    question_embedding = embedding_model.encode(
                        question,
                        normalize_embeddings=True
                    ).tolist()

                    results = collection.query(
                        query_embeddings=[question_embedding],
                        n_results=min(5, collection.count())
                    )

                    retrieved_chunks = results["documents"][0]
                    distances = results["distances"][0]
                    metadatas = results["metadatas"][0]

                    relevant_sources = []

                    for chunk, distance, metadata in zip(
                        retrieved_chunks, distances, metadatas
                    ):
                        if distance < 0.75:
                            relevant_sources.append({
                                "name": metadata.get(
                                    "source", "Unknown file"
                                ),
                                "text": chunk
                            })

                if not relevant_sources:
                    answer = (
                        "I couldn't find relevant information in "
                        "your documents. Try another question."
                    )
                    st.info(answer)

                else:
                    context_parts = []

                    for source in relevant_sources:
                        context_parts.append(
                            f"Source: {source['name']}\n"
                            f"Content: {source['text']}"
                        )

                    context = "\n\n".join(context_parts)

                    prompt = f"""
You are a helpful document question-answering assistant.

Answer using only the supplied document context.
Do not invent facts.
If the answer is not available in the context, say so.
Mention source filenames when useful.

Document context:
{context}

Question:
{question}

Answer:
"""

                    llm = OpenAI(api_key=api_key)

                    response = llm.responses.create(
                        model="gpt-6-luna",
                        input=prompt
                    )

                    answer = response.output_text
                    st.markdown(answer)

                    with st.expander("📖 View retrieved sources"):
                        for source in relevant_sources:
                            st.write(f"File: {source['name']}")
                            st.write(source["text"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": relevant_sources
                })

            except Exception as error:
                answer = f"Something went wrong: {error}"
                st.error(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })