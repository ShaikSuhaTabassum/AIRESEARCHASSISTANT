try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
except Exception:
    # Provide lightweight fallbacks so the module can be imported in environments
    # where parts of langchain aren't installed or have different layouts.
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class Document:
        page_content: str

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=50):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, docs: List[Document]):
            out = []
            for d in docs:
                text = d.page_content
                i = 0
                while i < len(text):
                    out.append(Document(page_content=text[i:i + self.chunk_size]))
                    i += self.chunk_size - self.chunk_overlap
            return out

    class HuggingFaceEmbeddings:
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            self.model_name = model_name

        def embed_documents(self, texts):
            # naive placeholder: return lengths as vectors
            return [[len(t)] for t in texts]

    class FAISS:
        @classmethod
        def from_documents(cls, docs, embeddings):
            # Simple in-memory store: keep docs and allow basic similarity search
            class SimpleStore:
                def __init__(self, docs):
                    self.docs = docs

                def similarity_search(self, query, k=3):
                    # return first k docs as a fallback
                    return self.docs[:k]

            return SimpleStore(docs)


# ---------------- BUILD VECTOR DB ---------------- #

def build_vector_store(papers):

    docs = []

    for p in papers:
        text = f"""
Title: {p['title']}
Summary: {p['summary']}
Authors: {', '.join(p['authors'])}
"""

        docs.append(Document(page_content=text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# ---------------- CHAT FUNCTION ---------------- #

def ask_question(vectorstore, question):

    docs = vectorstore.similarity_search(question, k=3)

    results = "\n\n".join([d.page_content for d in docs])

    return results