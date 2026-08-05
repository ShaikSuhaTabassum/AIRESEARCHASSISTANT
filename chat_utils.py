from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def ask_question(vectorstore, question):

    docs = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )

    prompt = f"""
You are an AI Research Assistant.

Answer ONLY using the research papers below.

Research Papers:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content