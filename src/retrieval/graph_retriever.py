import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize Neo4j
URI = os.getenv("NEO4J_URI").replace("neo4j+s://", "neo4j+ssc://")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# ─────────────────────────────────────────
# RETRIEVER 1: Graph Retriever
# ─────────────────────────────────────────
def graph_retriever(question):
    """Search Neo4j knowledge graph for relevant entities"""
    print("\n🔍 Graph Retriever searching Neo4j...")

    with driver.session() as session:
        # Find entities related to keywords in question
        words = [w for w in question.lower().split() if len(w) > 3]

        results = []
        for word in words[:3]:  # Check top 3 keywords
            result = session.run("""
                MATCH (e:Entity)-[r]->(e2:Entity)
                WHERE toLower(e.name) CONTAINS $keyword
                   OR toLower(e2.name) CONTAINS $keyword
                RETURN e.name as source, r.type as relation, e2.name as target
                LIMIT 10
            """, keyword=word)

            for record in result:
                results.append(
                    f"{record['source']} --[{record['relation']}]--> {record['target']}"
                )

    # Remove duplicates
    results = list(set(results))
    print(f"  Found {len(results)} graph connections!")
    return "\n".join(results) if results else "No graph connections found"


# ─────────────────────────────────────────
# RETRIEVER 2: Vector Retriever
# ─────────────────────────────────────────
def build_vector_store(chunks):
    """Build ChromaDB vector store from document chunks"""
    print("\n📦 Building vector store...")
    embeddings = FakeEmbeddings(size=384)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    print("  ✅ Vector store ready!")
    return vectorstore

def vector_retriever(vectorstore, question):
    """Search ChromaDB for similar text chunks"""
    print("\n🔍 Vector Retriever searching ChromaDB...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    results = "\n\n".join([doc.page_content for doc in docs])
    print(f"  Found {len(docs)} relevant chunks!")
    return results


# ─────────────────────────────────────────
# ANSWER GENERATOR
# ─────────────────────────────────────────
def generate_answer(question, graph_context, vector_context):
    """Generate final answer using both graph and vector context"""
    print("\n🤖 Generating answer with Groq LLM...")

    prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant. Answer the question using the context below.

GRAPH CONTEXT (relationships between concepts):
{graph_context}

DOCUMENT CONTEXT (relevant text passages):
{vector_context}

Question: {question}

Give a clear, detailed answer based on the context above.
If the context doesn't contain enough information, say so honestly.
""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": question,
        "graph_context": graph_context,
        "vector_context": vector_context
    })
    return answer


# ─────────────────────────────────────────
# MAIN GRAPH RAG PIPELINE
# ─────────────────────────────────────────
def graph_rag_pipeline(question, vectorstore):
    """Complete Graph RAG pipeline"""
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}")

    # Step 1: Graph retrieval
    graph_context = graph_retriever(question)

    # Step 2: Vector retrieval
    vector_context = vector_retriever(vectorstore, question)

    # Step 3: Generate answer
    answer = generate_answer(question, graph_context, vector_context)

    print(f"\n{'='*50}")
    print(f"ANSWER:\n{answer}")
    print(f"{'='*50}\n")

    return answer


# ─────────────────────────────────────────
# RUN IT
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("E:\\graph-rag-project")
    from src.ingestion.document_loader import load_documents

    # Load documents and build vector store
    chunks = load_documents()
    vectorstore = build_vector_store(chunks)

    # Ask questions!
    questions = [
        "What is an AI agent?",
        "How does planning work in AI agents?",
        "What tools do AI agents use?"
    ]

    for question in questions:
        graph_rag_pipeline(question, vectorstore)

    driver.close()