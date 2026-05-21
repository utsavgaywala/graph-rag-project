import os
import sys
sys.path.append("E:\\graph-rag-project")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Initialize everything
llm = ChatGroq(model="llama-3.1-8b-instant")
URI = os.getenv("NEO4J_URI").replace("neo4j+s://", "neo4j+ssc://")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# ─────────────────────────────────────────
# GRAPH RAG PROMPT
# ─────────────────────────────────────────
graph_rag_prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant specializing in AI research.
Answer the question using ONLY the context provided below.

KNOWLEDGE GRAPH CONTEXT:
(These are connected facts from a knowledge graph)
{graph_context}

DOCUMENT CONTEXT:
(These are relevant passages from the research paper)
{vector_context}

Question: {question}

Instructions:
- Use the knowledge graph context to understand relationships
- Use the document context for detailed explanations  
- Give a clear, structured answer
- If context is insufficient, say "I need more context"
- Do NOT make up information

Answer:
""")

def graph_retriever(question):
    """Retrieve from Neo4j knowledge graph"""
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
    return "\n".join(results) if results else "No graph context found"

def vector_retriever(vectorstore, question):
    """Retrieve from ChromaDB vector store"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])

def generate_final_answer(question, graph_context, vector_context):
    """Generate final answer using Graph RAG"""
    chain = graph_rag_prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": question,
        "graph_context": graph_context,
        "vector_context": vector_context
    })

def graph_rag(question, vectorstore):
    """Complete Graph RAG pipeline"""
    print(f"\n{'='*60}")
    print(f"🤔 Question: {question}")
    print(f"{'='*60}")

    # Retrieve from both sources
    print("🔍 Searching knowledge graph...")
    graph_context = graph_retriever(question)

    print("🔍 Searching document chunks...")
    vector_context = vector_retriever(vectorstore, question)

    # Generate answer
    print("🤖 Generating answer...")
    answer = generate_final_answer(question, graph_context, vector_context)

    print(f"\n✅ ANSWER:\n{answer}")
    print(f"{'='*60}\n")
    return answer

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    from src.ingestion.document_loader import load_documents

    print("🚀 Starting Graph RAG System...")
    print("Loading documents...")
    chunks = load_documents()

    print("Building vector store...")
    embeddings = FakeEmbeddings(size=384)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("✅ Vector store ready!")

    # Test questions
    questions = [
        "What is an AI agent and how does it work?",
        "How does planning work in AI agents?",
        "What is the difference between short term and long term memory in AI agents?",
    ]

    for question in questions:
        graph_rag(question, vectorstore)

    driver.close()
    print("✅ Graph RAG pipeline complete!")