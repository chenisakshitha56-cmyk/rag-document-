import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mini RAG",
    page_icon="📚"
)

st.title("📚 Mini RAG: Document Store + Retrieval")
st.caption("PDF ➜ Chunks ➜ Embeddings ➜ ChromaDB ➜ Retrieval ➜ Ollama")


# -----------------------------
# Load Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# -----------------------------
# ChromaDB Setup
# -----------------------------
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.header("⚙️ Settings")

ollama_model = st.sidebar.text_input(
    "Ollama model",
    "llama3.2"
)

chunk_size = st.sidebar.slider(
    "Chunk size",
    200,
    1500,
    500,
    100
)

top_k = st.sidebar.slider(
    "Chunks to retrieve",
    1,
    5,
    3
)


# -----------------------------
# Build Document Store
# -----------------------------
st.header("1️⃣ Build Document Store")

uploaded_file = st.file_uploader(
    "Upload a text-based PDF",
    type=["pdf"]
)


if uploaded_file and st.button("⬇️ Process & Store PDF"):

    reader = PdfReader(uploaded_file)

    text = ""

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # Check whether PDF contains readable text
    if not text.strip():

        st.error(
            "No readable text was found. "
            "Try uploading a text-based PDF."
        )

        st.stop()

    # -----------------------------
    # Create Chunks
    # -----------------------------
    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    # -----------------------------
    # Generate Embeddings
    # -----------------------------
    with st.spinner("Generating embeddings..."):

        embeddings = model.encode(chunks)

    # -----------------------------
    # Remove Existing Documents
    # -----------------------------
    existing = collection.get()

    if existing["ids"]:

        collection.delete(
            ids=existing["ids"]
        )

    # -----------------------------
    # Store Documents in ChromaDB
    # -----------------------------
    collection.add(

        ids=[
            f"chunk_{i}"
            for i in range(len(chunks))
        ],

        documents=chunks,

        embeddings=embeddings.tolist(),

        metadatas=[
            {
                "source": uploaded_file.name,
                "chunk": i
            }

            for i in range(len(chunks))
        ]
    )

    st.success(
        f"✅ Stored {len(chunks)} chunks in ChromaDB."
    )

    # -----------------------------
    # Preview Chunks
    # -----------------------------
    with st.expander("Preview stored chunks"):

        for i, chunk in enumerate(chunks[:5]):

            st.write(
                f"**Chunk {i + 1}:**"
            )

            st.write(chunk)

            st.divider()


# -----------------------------
# Ask Questions
# -----------------------------
st.header("2️⃣ Ask Questions")

question = st.text_input(
    "Ask a question about your uploaded document"
)


if st.button("🔍 Retrieve & Answer"):

    # Check question
    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # Check document store
    count = collection.count()

    if count == 0:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()

    # -----------------------------
    # Retrieve Relevant Chunks
    # -----------------------------
    with st.spinner("Searching document..."):

        question_embedding = model.encode(
            [question]
        )[0]

        results = collection.query(

            query_embeddings=[
                question_embedding.tolist()
            ],

            n_results=min(top_k, count)
        )

    retrieved_chunks = results["documents"][0]

    # -----------------------------
    # Display Retrieved Chunks
    # -----------------------------
    st.subheader("📄 Retrieved Chunks")

    for i, retrieved_chunk in enumerate(
        retrieved_chunks
    ):

        with st.expander(
            f"Retrieved Chunk {i + 1}"
        ):

            st.write(retrieved_chunk)

    # -----------------------------
    # Create Context
    # -----------------------------
    context = "\n\n".join(
        retrieved_chunks
    )

    # -----------------------------
    # Create Prompt
    # -----------------------------
    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY
the context provided below.

If the answer is not present in the context,
say:

"I could not find the answer in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    # -----------------------------
    # Generate Answer using Ollama
    # -----------------------------
    with st.spinner(
        f"Generating answer with {ollama_model}..."
    ):

        try:

            response = ollama.chat(

                model=ollama_model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response[
                "message"
            ][
                "content"
            ]

            st.subheader("🤖 Answer")

            st.write(answer)

        except Exception as e:

            st.error(
                f"""
Could not connect to Ollama.

Make sure Ollama is running and the model
'{ollama_model}' is installed.

Error: {e}
"""
            )