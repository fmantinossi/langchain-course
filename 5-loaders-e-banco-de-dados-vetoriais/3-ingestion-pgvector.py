import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

for var in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(var):
        raise RuntimeError(f"Missing environment variable: {var}")

current_dir = Path(__file__).parent
pdf_path = current_dir / "gpt5.pdf"

docs = PyPDFLoader(str(pdf_path)).load()

splits = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100,
    add_start_index = False
).split_documents(docs)

if not splits:
    raise SystemExit(1)

#enriched_document = [
#    Document(
#        page_content=d.page_content,
#        metadata={k: v for k,v in d.metadata.items() if v not in ("", None)}
#    )
#    for d in splits
#]

enriched_document = []
for d in splits:
    metadata = {k: v for k, v in d.metadata.items() if v not in ("", None)}
    new_document = Document(
        page_content=d.page_content,
        metadata=metadata
    )
    enriched_document.append(new_document)

ids = [f"doc-{i}" for i in range(len(enriched_document))]

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL","text-embedding-3-small"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

store.add_documents(documents=enriched_document, ids=ids)