from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

result = embeddings.embed_query(
    "Artificial Intelligence in Healthcare"
)

print("Embedding dimension:", len(result))
print("✅ Embedding works!")