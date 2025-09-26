import os
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

for var in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(var):
        raise RuntimeError(f"Missing environment variable: {var}")
    
query = "Me diga mais informações sobre a performance e validações do GPT-5 em comparação ao GPT-4."

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL","text-embedding-3-small"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

results = store.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results,start=1):
    print("-"*50)
    print(f"Resultado {i} (score: {score:.2f}): ")
    print("="*50)
    print("Text\n")
    print(doc.page_content.strip())
    print("\nMetadados:\n")
    for k, v in doc.metadata.items():
        print(f"{k}: {v}")
          