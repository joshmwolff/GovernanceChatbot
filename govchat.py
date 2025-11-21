import streamlit as st
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


# Set API key securely
openai.api_key = st.secrets["openai_api_key"]
doc_path = "DataGov.docx"
        doc = docx.Document(doc_path)

# ---------------------------
# Load and parse the Word document
# ---------------------------
@st.cache_resource
def load_document():
    doc = doc
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return full_text

# ---------------------------
# Create or load vector store
# ---------------------------
@st.cache_resource
def build_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300
    )
    chunks = text_splitter.split_text(text)

embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model="text-embedding-3-large"   # optional but recommended
)
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./vectorstore"
    )
    vectordb.persist()
    return vectordb

# ---------------------------
# Retrieve and answer
# ---------------------------
def answer_query(query, vectordb):
    docs = vectordb.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You must answer strictly and only from the document text below.

Document context:
-----------------
{context}
-----------------

User question: {query}

If the document does not contain the answer, say so.
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

    with st.spinner("Loading document and building embeddings..."):
        text = load_document()
        vectordb = build_vectorstore(text)

    user_query = st.text_input("Ask a question about the guide:")

    if user_query:
        with st.spinner("Retrieving answer..."):
            answer = answer_query(user_query, vectordb)
        st.write("### Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
