import os
import sys
sys.path.append("E:\\graph-rag-project")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from neo4j import GraphDatabase
import uvicorn

load_dotenv()

# ─────────────────────────────────────────
# Initialize everything
# ─────────────────────────────────────────
app = FastAPI(
    title="Graph RAG API",
    description="Industry-level Graph RAG system by Utsav Gaywala",
    version="1.0.0"
)

llm = ChatGroq(model="llama-3.1-8b-instant")
URI = os.getenv("NEO4J_URI").replace("neo4j+s://", "neo4j+ssc://")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# Build vector store on startup
from src.ingestion.document_loader import load_documents
chunks = load_documents()
embeddings = FakeEmbeddings(size=384)
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

# ─────────────────────────────────────────
# Request/Response models
# ─────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    graph_connections: int
    vector_chunks: int

# ─────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────
def graph_retriever(question):
    with driver.session() as session:
        words = [w for w in question.lower().split() if len(w) > 3]
        results = []
        for word in words[:5]:
            result = session.run("""
                MATCH (e:Entity)-[r]->(e2:Entity)
                WHERE toLower(e.name) CONTAINS $keyword
                   OR toLower(e2.name) CONTAINS $keyword
                RETURN e.name as source, r.type as relation, e2.name as target
                LIMIT 15
            """, keyword=word)
            for record in result:
                results.append(
                    f"{record['source']} --[{record['relation']}]--> {record['target']}"
                )
    results = list(set(results))
    return results

def vector_retriever(question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]

def generate_answer(question, graph_context, vector_context):
    prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant. Answer the question using the context below.

KNOWLEDGE GRAPH CONTEXT:
{graph_context}

DOCUMENT CONTEXT:
{vector_context}

Question: {question}

Give a clear, structured answer based on the context.
""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": question,
        "graph_context": "\n".join(graph_context),
        "vector_context": "\n\n".join(vector_context)
    })

# ─────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Graph RAG API is running!",
        "author": "Utsav Gaywala",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        # Retrieve from both sources
        graph_context = graph_retriever(request.question)
        vector_context = vector_retriever(request.question)

        # Generate answer
        answer = generate_answer(
            request.question,
            graph_context,
            vector_context
        )

        return AnswerResponse(
            question=request.question,
            answer=answer,
            graph_connections=len(graph_context),
            vector_chunks=len(vector_context)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/stats")
def graph_stats():
    with driver.session() as session:
        nodes = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
    return {
        "total_nodes": nodes,
        "total_relationships": rels
    }

# ─────────────────────────────────────────
# Run server
# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)