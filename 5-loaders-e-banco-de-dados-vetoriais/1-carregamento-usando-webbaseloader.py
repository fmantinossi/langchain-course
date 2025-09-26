from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = WebBaseLoader("https://www.langchain.com/").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.page_content)
    print("\n---\n")