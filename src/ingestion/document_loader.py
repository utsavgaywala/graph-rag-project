import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4

load_dotenv()

def load_documents():
    print("Loading research paper...")
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title")))
    )
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} document(s)")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    chunks = load_documents()
    print(f"\nFirst chunk preview:")
    print(chunks[0].page_content[:300])