from arxiv_helper import get_arxiv_papers
from vector_db import build_vector_store

papers = get_arxiv_papers(
    "Artificial Intelligence in Healthcare",
    5
)

db = build_vector_store(papers)

print("✅ Vector DB created successfully!")
