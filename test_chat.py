from vector_db import load_vector_store
from chat_utils import ask_question

db = load_vector_store()

answer = ask_question(
    db,
    "What is Explainable AI in Healthcare?"
)

print(answer)