import streamlit as st
from openai import OpenAI
from docx import Document
import chromadb
import hashlib

# Load key from Streamlit Secrets
client = OpenAI(api_key=st.secrets["openai_api_key"])


# ---------------------------
# Load the Word document
# ---------------------------
@st.cache_resource
def load_document():
    doc = Document("DataGov.docx")
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text


# ---------------------------
# Simple text chunker
# ---------------------------
def chunk_text(text, chunk_size=1800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------------------
# Build Chroma vectorstore
# ---------------------------
@st.cache_resource
def build_vectordb(text):
    chroma_client = chromadb.PersistentClient(path="./vectorstore")

    # Create or load collection
    try:
        collection = chroma_client.create_collection(
            name="datagov",
            metadata={"hnsw:space": "cosine"}
        )
    except:
        collection = chroma_client.get_collection("datagov")

    # Only embed once
    if collection.count() == 0:
        chunks = chunk_text(text)

        ids = []
        docs = []
        embeddings = []

        for chunk in chunks:
            cid = hashlib.md5(chunk.encode()).hexdigest()
            ids.append(cid)
            docs.append(chunk)

            emb = client.embeddings.create(
                model="text-embedding-3-large",
                input=chunk
            ).data[0].embedding

            embeddings.append(emb)

        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings
        )

    return collection


# ---------------------------
# Query & answer
# ---------------------------
def answer_query(query, collection):
    # Embed question
    q_emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding

    # Retrieve chunks
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=8
    )

    docs = results["documents"][0]
    context = "\n\n".join(docs)

    prompt = f"""
    You are a helpful assistant grounded ONLY in the following document excerpts:

    {context}

    User question: {query}

    If the document does not contain the answer, say:
    "The document does not appear to include that information."
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content


# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.title("Data Governance Starter Guide – Document Chat")

    with st.spinner("Loading and indexing document..."):
        text = load_document()
        vectordb = build_vectordb(text)

    query = st.text_input("Ask a question about the guide:")

    if query:
        with st.spinner("Retrieving answer..."):
            answer = answer_query(query, vectordb)
        st.markdown("### Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
